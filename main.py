import os; import json;
from ctypes.macholib import framework

import tiktoken; from transformers import AutoTokenizer

def findMatchingTokenizer(calibTxt, calibIDs):
    tiktokenEncodings = [
        "cl100k_base",
        "p50k_base",
        "r50k_base",
        "o200k_base",
        "p50k_edit",
        "gpt2"
    ]

    for encoding in tiktokenEncodings:
        try:
            encoder = tiktoken.get_encoding(encoding)
            tokenized = encoder.encode(calibTxt)
            if tokenized == calibIDs:
                return ("tiktoken", encoding, encoder)
        except Exception as e:
            print(f"Error with encoding {encoding}: {e}")
            continue
    hfModels = [
        "gpt2",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "Qwen/Qwen2.5-7B",
        "google/flan-t5-base",
        "bert-base-uncased",
        "roberta-base"
    ]
    for model in hfModels:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            tokenized = tokenizer.encode(calibTxt, add_special_tokens=False)
            if tokenized == calibIDs:
                return ("transformers", model, tokenizer)
        except Exception as e:
            print(f"Error with model {model}: {e}")
            continue

    raise ValueError("No matching tokenizer found for the provided calibration text and IDs.")

def decodeTokens(tokenizerType, tokenizerObj, challengeIDs):
    if tokenizerType == "tiktoken":
        return tokenizerObj.decode(challengeIDs)
    elif tokenizerType == "transformers":
        return tokenizerObj.decode(challengeIDs, skip_special_tokens=False) # Keep special tokens if any
    raise ValueError("Unknown tokenizer type.")

def main():
    inputFN = "test-input.json"
    outputFN = "test-output.json"

    if not os.path.exists(inputFN) and os.path.exists(outputFN):
        inputFN = "train-input.json"
        outputFN = "test-output.json"
    with open(inputFN, "r", encoding="utf-8") as f:
        data = json.load(f)

    calibrationTxt = data["calibration"]["text"]
    calibrationIDs = data["calibration"]["token_ids"]
    challangeIDs = data["challenge"]["token_ids"]

    print(f"[*] identifing tokenizer using CalTXT ({len(calibrationTxt)} chars)")
    framework, tokenizerID, tokenizerObj = findMatchingTokenizer(calibrationTxt, calibrationIDs)
    print(f"[+] identified tokemizer: [{framework}] {tokenizerID}")

    decodedMsg = decodeTokens(framework, tokenizerObj, challangeIDs)
    print(f"[+] decoded message: [{decodedMsg}]")

    outputDat = {
        "tokenizer_id": tokenizerID,
        "decoded_msg": decodedMsg
    }

    with open(outputFN, "w", encoding="utf-8") as f:
        json.dump(outputDat, f, indent=2)
    print(f"[+] output written to file: [{outputFN}]")

if __name__ == "__main__":
    main()