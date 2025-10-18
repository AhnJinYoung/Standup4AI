#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import random
import re
from typing import List, Dict, Any

from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer


BASE_MODEL = "openai/gpt-oss-20b"
RANDOM_SEED = 42


def clean_text(t: str) -> str:
    """줄바꿈/공백 중심의 최소 정리. 무대지시어/기호는 유지."""
    if not isinstance(t, str):
        return ""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # 줄 끝 공백 제거
    t = re.sub(r"[ \t]+\n", "\n", t)
    # 과도한 빈 줄 2개로 축소
    t = re.sub(r"\n{3,}", "\n\n", t)
    # HTML 잔여물(아주 가볍게)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.IGNORECASE)
    return t.strip()


def pick(ex: Dict[str, Any], candidates: List[str], default: str = "") -> str:
    for k in candidates:
        v = ex.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def split_bits_by_markers(text: str) -> List[str]:
    """
    코미디 '비트' 단위 후보로 문단/무대지시어를 기준 분절.
    큰 빈줄 or [laughter]/[applause]에서 끊고, 텍스트 블록만 반환.
    """
    if not text:
        return []
    # split할 때 구분자도 캡쳐되므로 후처리 필요
    parts = re.split(r"(\n{2,}|\[laughter\]|\[applause\])", text, flags=re.IGNORECASE)
    bits = []
    buf = ""
    for p in parts:
        if p is None:
            continue
        if re.fullmatch(r"\n{2,}", p) or p.lower() in ("[laughter]", "[applause]"):
            # 경계 도달 → 지금까지 버퍼를 하나의 비트로
            if buf.strip():
                bits.append(buf.strip())
                buf = ""
            # 구분 자체도 의미가 있으므로 한 줄로 남겨도 됨 (선택)
            # 여기서는 구분 토큰을 다음 비트의 앞 힌트로 붙이지 않고 건너뜀
        else:
            buf += (("\n\n" if buf else "") + p)
    if buf.strip():
        bits.append(buf.strip())
    # 너무 잘게 쪼개졌다면 합치는 로직을 넣을 수도 있지만 우선 단순 반환
    return [b for b in bits if b.strip()]


def render_token_len(tokenizer, messages: List[Dict[str, str]]) -> int:
    """gpt-oss Harmony 채팅 템플릿을 적용한 후의 토큰 길이."""
    ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_tensors="pt",
    )
    return int(ids.shape[-1])


def hard_slice_by_tokens(tokenizer, dev_user_msgs, text: str, max_len: int) -> List[List[Dict[str, str]]]:
    """
    비트 하나가 너무 길어서 통째로 못 넣을 때 토큰 기준 하드 슬라이스 (최소 침습).
    """
    encoded = tokenizer(text, return_tensors="pt").input_ids[0]
    step = max_len - 128  # 여유 버퍼
    out = []
    for i in range(0, len(encoded), step):
        sub = tokenizer.decode(encoded[i : i + step])
        out.append(dev_user_msgs + [{"role": "assistant", "content": sub}])
    return out


def chunk_with_overlap(
    tokenizer,
    base_dev: str,
    user_prompt: str,
    body: str,
    max_len: int = 4096,
    overlap: int = 512,
) -> List[List[Dict[str, str]]]:
    """
    - 비트 단위로 쌓아가며 max_len 넘치면 청크 확정
    - 청크 간 오버랩 토큰(assistant 텍스트의 뒤쪽)을 남겨 콜백/맥락 보존
    """
    dev_user = [
        {"role": "developer", "content": base_dev},
        {"role": "user", "content": user_prompt},
    ]
    bits = split_bits_by_markers(body)
    out = []
    cur_segments = []

    def as_msgs(txt: str):
        return dev_user + [{"role": "assistant", "content": txt}]

    for b in bits:
        trial_txt = "\n\n".join(cur_segments + [b]).strip()
        if render_token_len(tokenizer, as_msgs(trial_txt)) <= max_len:
            cur_segments.append(b)
            continue

        # 넘친 경우 → 지금까지를 청크로 확정
        if cur_segments:
            cur_txt = "\n\n".join(cur_segments)
            out.append(as_msgs(cur_txt))

            # 오버랩 생성 (assistant 텍스트 뒤에서 overlap 토큰 유지)
            cur_ids = tokenizer(cur_txt, return_tensors="pt").input_ids[0]
            keep_ids = cur_ids[-overlap:] if overlap > 0 and len(cur_ids) > overlap else cur_ids
            keep_txt = tokenizer.decode(keep_ids)

            # 겹침 + 새 비트로 다음 청크 시작
            cur_segments = [keep_txt, b]
            # 혹시 시작부터 또 넘치면 하드 슬라이스
            if render_token_len(tokenizer, as_msgs("\n\n".join(cur_segments))) > max_len:
                out.extend(hard_slice_by_tokens(tokenizer, dev_user, "\n\n".join(cur_segments), max_len))
                cur_segments = []
        else:
            # 비트 하나가 너무 클 때 하드 슬라이스
            out.extend(hard_slice_by_tokens(tokenizer, dev_user, b, max_len))

    if cur_segments:
        out.append(as_msgs("\n\n".join(cur_segments)))

    return out


def build_messages_record(title: str, comic: str, body: str, add_meta: bool = True) -> Dict[str, Any]:
    """Developer/User/Assistant 메시지 블록과 해시 생성."""
    # Developer 프롬프트 (메타 포함 가능)
    dev = (
        "You are a stand-up comedian. When asked to perform, reply with a stand-up monologue. "
        "Preserve line breaks and stage directions like [laughter] or [applause]."
    )
    if add_meta:
        dev += f"\n[metadata] title={title} | comedian={comic}"

    user = f"Please perform a stand-up set.\nTitle: {title}\nComedian: {comic}"
    body_clean = clean_text(body)
    h = hashlib.md5(" ".join(body_clean.split()).lower().encode("utf-8")).hexdigest()

    messages = [
        {"role": "developer", "content": dev},
        {"role": "user", "content": user},
        {"role": "assistant", "content": body_clean},
    ]
    return {"messages": messages, "text_hash": h}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=str, default="zachgitt/comedy-transcripts")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--overlap", type=int, default=384)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--push-to-hub", type=str, default=None, help="e.g., napalna/comedy-transcripts-clean")
    parser.add_argument("--out-dir", type=str, default="./prep_out")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    print(f"[INFO] Loading dataset: {args.dataset_id} ({args.split})")
    raw = load_dataset(args.dataset_id, split=args.split)

    print("[INFO] Loading tokenizer:", BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)

    title_keys = ["title", "album", "show"]
    comic_keys = ["comic", "comedian", "author", "speaker"]
    text_keys = ["transcript", "text", "content", "body"]

    # 1차 가공: 메시지 레코드 생성 + 중복 제거
    seen = set()
    base_records = []
    for ex in raw:
        t = pick(ex, text_keys, "")
        if not t or len(t.strip()) < 50:
            continue
        title = pick(ex, title_keys, "Untitled Set")
        comic = pick(ex, comic_keys, "Unknown Comedian")

        rec = build_messages_record(title, comic, t, add_meta=True)
        if rec["text_hash"] in seen:
            continue
        seen.add(rec["text_hash"])
        base_records.append(rec)

    print(f"[INFO] Base records (deduped): {len(base_records)}")

    # 2차 가공: max_length/overlap 기준 청크 분할
    final_records = []
    for rec in base_records:
        dev = rec["messages"][0]["content"]
        user = rec["messages"][1]["content"]
        body = rec["messages"][2]["content"]

        # 통째로 들어가면 자르지 않음
        if render_token_len(tok, rec["messages"]) <= args.max_length:
            final_records.append({"messages": rec["messages"]})
            continue

        # 비트 단위 + 오버랩 분할
        chunks = chunk_with_overlap(
            tokenizer=tok,
            base_dev=dev,
            user_prompt=user,
            body=body,
            max_len=args.max_length,
            overlap=args.overlap,
        )
        for c in chunks:
            final_records.append({"messages": c})

    if args.shuffle:
        random.shuffle(final_records)

    ds = Dataset.from_list(final_records)
    print(ds)

    # 저장/업로드
    if args.push_to_hub:
        print(f"[INFO] Pushing to hub: {args.push_to_hub}")
        ds.push_to_hub(args.push_to_hub)
    else:
        # 로컬 Parquet 저장
        out_path = os.path.join(args.out_dir, "comedy_messages.parquet")
        ds.to_parquet(out_path)
        print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()
