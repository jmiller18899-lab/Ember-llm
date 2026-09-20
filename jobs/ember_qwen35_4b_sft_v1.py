# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["torch==2.11.0", "transformers==5.17.0", "peft==0.20.0", "trl==1.13.0", "accelerate==1.15.0", "huggingface-hub==1.31.0", "trackio==0.37.1", "datasets==5.0.1"]
# ///
"""One conservative Qwen3.5-4B LoRA experiment; no deployment or automatic retries."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

MODEL = "Qwen/Qwen3.5-4B"
FRESH_NATURAL_CASES = json.loads("[{\"id\":\"fresh-natural-v2-01\",\"family\":\"greeting\",\"prompt\":\"You can call me Sienna. How would you greet me?\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Greet Sienna; do not claim to be Sienna.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-02\",\"family\":\"greeting\",\"prompt\":\"Hi there—can we talk for a minute?\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Respond naturally and offer to chat without invented context.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-03\",\"family\":\"drafting\",\"prompt\":\"Message Mateo: I found his notebook, and I can drop it off tomorrow.\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Draft to Mateo from the sender; retain found notebook and tomorrow.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-04\",\"family\":\"drafting\",\"prompt\":\"Polish this without dropping either point: 'The link has expired. Please request a new one.'\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Retain both expired link and request for a new link.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-05\",\"family\":\"drafting\",\"prompt\":\"A customer says, 'That fixed it, thank you!' Write a short response.\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Reply to the customer's thanks; do not draft a new unrelated thank-you.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-06\",\"family\":\"clarification\",\"prompt\":\"Could you make that clearer?\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Ask for the text or topic; there is no previous content.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-07\",\"family\":\"clarification\",\"prompt\":\"The train left at 2 PM. What time did it reach its destination?\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Cannot determine arrival without journey duration or arrival information.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-08\",\"family\":\"clarification\",\"prompt\":\"Have you emailed the invoice? This conversation has no email tool.\",\"answer\":null,\"scoring\":\"rubric\",\"rubric\":\"Do not claim an email was sent; acknowledge inability.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-09\",\"family\":\"arithmetic\",\"prompt\":\"There are 12 apples and 5 pears in a basket. We remove 4 apples. How many pieces of fruit are left? Give just the number.\",\"answer\":\"13\",\"scoring\":\"exact\",\"rubric\":\"Subtract removed apples from total fruit.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-10\",\"family\":\"arithmetic\",\"prompt\":\"A jar holds 20 coins. Add 6, then take away 9. Return the final count only.\",\"answer\":\"17\",\"scoring\":\"exact\",\"rubric\":\"Apply both operations.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-11\",\"family\":\"arithmetic\",\"prompt\":\"Four shelves hold 3 books each. How many books altogether? Number only.\",\"answer\":\"12\",\"scoring\":\"exact\",\"rubric\":\"Multiply four by three.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-12\",\"family\":\"copy\",\"prompt\":\"Remove only the surrounding braces and return the contents: {ID=qZ_003; result=error}\",\"answer\":\"ID=qZ_003; result=error\",\"scoring\":\"exact\",\"rubric\":\"Remove delimiters only; preserve case and punctuation.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-13\",\"family\":\"copy\",\"prompt\":\"Print exactly this sentence: No warning remains; don't restart.\",\"answer\":\"No warning remains; don't restart.\",\"scoring\":\"exact\",\"rubric\":\"Do not classify or change punctuation.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-14\",\"family\":\"extraction\",\"prompt\":\"Jin owns the green bicycle; Rosa owns the red bicycle. Give only the owner of the red one.\",\"answer\":\"Rosa\",\"scoring\":\"exact\",\"rubric\":\"Choose correct owner.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-15\",\"family\":\"classification\",\"prompt\":\"Use only success, warning, or error: The backup finished, but its integrity check failed and restoration is impossible.\",\"answer\":\"error\",\"scoring\":\"exact\",\"rubric\":\"Failure preventing restoration overrides completed copying.\",\"suite\":\"fresh_natural_v2\"},{\"id\":\"fresh-natural-v2-16\",\"family\":\"classification\",\"prompt\":\"Choose success, warning, or error, and nothing else: A maintenance delay is expected; all functions still work, but attention is needed.\",\"answer\":\"warning\",\"scoring\":\"exact\",\"rubric\":\"Risk requiring attention despite availability.\",\"suite\":\"fresh_natural_v2\"}]")
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
REPO = "Jmiller18899/ember-qwen3.5-4b-sft-v1"
SYSTEM = "You are Ember. Answer the current request directly and concisely. Preserve supplied facts and follow the requested format."
REGRESSION_CASES = json.loads("[{\"exact_reference\":null,\"family\":\"greeting\",\"id\":\"natural-v1-01\",\"prompt\":\"Hey Ember, how's it going?\",\"rubric\":\"Respond naturally to the greeting; no invented person, completed task, or thank-you template.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"greeting\",\"id\":\"natural-v1-02\",\"prompt\":\"Morning! I'm just checking whether you can chat.\",\"rubric\":\"Acknowledge greeting and chat request.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"greeting\",\"id\":\"natural-v1-03\",\"prompt\":\"I'm Rowan. Say hi using my name, in one short sentence.\",\"rubric\":\"Greet Rowan without claiming a previous relationship.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"greeting\",\"id\":\"natural-v1-04\",\"prompt\":\"Hello. You don't know my name yet.\",\"rubric\":\"Greet without inventing a name.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"greeting\",\"id\":\"natural-v1-05\",\"prompt\":\"Please welcome our new teammate, Amara. Keep it warm and brief.\",\"rubric\":\"Address Amara with a welcome; do not thank or answer as Amara.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"pR-008_Zx\",\"family\":\"copy\",\"id\":\"natural-v1-06\",\"prompt\":\"Send back only this code, keeping every character: pR-008_Zx\",\"rubric\":\"Exact preservation.\",\"answer\":\"pR-008_Zx\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"The upload failed, but the original file is safe.\",\"family\":\"copy\",\"id\":\"natural-v1-07\",\"prompt\":\"Repeat this sentence verbatim: The upload failed, but the original file is safe.\",\"rubric\":\"Do not classify the quoted error.\",\"answer\":\"The upload failed, but the original file is safe.\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"warning / SUCCESS / error?!\",\"family\":\"copy\",\"id\":\"natural-v1-08\",\"prompt\":\"Copy exactly, including the punctuation: warning / SUCCESS / error?!\",\"rubric\":\"Do not interpret status words.\",\"answer\":\"warning / SUCCESS / error?!\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"Total: -$38.07; due 07/04.\",\"family\":\"copy\",\"id\":\"natural-v1-09\",\"prompt\":\"Return only the line between brackets: [Total: -$38.07; due 07/04.]\",\"rubric\":\"Preserve sign, punctuation, and date.\",\"answer\":\"Total: -$38.07; due 07/04.\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"/srv/Ember-v2/cache_a.json\",\"family\":\"copy\",\"id\":\"natural-v1-10\",\"prompt\":\"I need this path unchanged and nothing else: /srv/Ember-v2/cache_a.json\",\"rubric\":\"Case-sensitive copy.\",\"answer\":\"/srv/Ember-v2/cache_a.json\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"29\",\"family\":\"extraction\",\"id\":\"natural-v1-11\",\"prompt\":\"Priya packed 14 boxes; Theo packed 29. How many did Theo pack? Answer with the number only.\",\"rubric\":\"Select correct person.\",\"answer\":\"29\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"11:35 AM\",\"family\":\"extraction\",\"id\":\"natural-v1-12\",\"prompt\":\"The appointment was moved from 8:20 AM to 11:35 AM. What is the new time? Return only the time.\",\"rubric\":\"Distinguish old from new.\",\"answer\":\"11:35 AM\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"VN-62\",\"family\":\"extraction\",\"id\":\"natural-v1-13\",\"prompt\":\"Order KX-41 was canceled. Order VN-62 is confirmed. Give only the confirmed order ID.\",\"rubric\":\"Ignore canceled distractor.\",\"answer\":\"VN-62\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"007\",\"family\":\"extraction\",\"id\":\"natural-v1-14\",\"prompt\":\"From 'owner=Sal; count=007; state=ready', extract count exactly, including leading zeros.\",\"rubric\":\"Preserve leading zeros.\",\"answer\":\"007\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"Tess\",\"family\":\"extraction\",\"id\":\"natural-v1-15\",\"prompt\":\"A note says: approved by Niko, reviewed by Tess. Who reviewed it? Name only.\",\"rubric\":\"Correct role.\",\"answer\":\"Tess\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"drafting\",\"id\":\"natural-v1-16\",\"prompt\":\"Can you write a quick text to Delia thanking her for bringing the spare charger?\",\"rubric\":\"Draft thanks addressed to Delia; retain spare charger; do not reply you're welcome.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"drafting\",\"id\":\"natural-v1-17\",\"prompt\":\"Omar wrote, 'Thanks for helping me move.' Give me a friendly reply I can send.\",\"rubric\":\"Reply to Omar's thanks, not new thanks for invented work.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"drafting\",\"id\":\"natural-v1-18\",\"prompt\":\"Tell Felix I'm running ten minutes late and ask if we can meet at 3:40 instead.\",\"rubric\":\"Draft from speaker to Felix, preserve ten minutes and 3:40.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"drafting\",\"id\":\"natural-v1-19\",\"prompt\":\"Make this less blunt without changing the meaning: 'Your attachment is missing. Send it again.'\",\"rubric\":\"Polite rewrite requesting missing attachment, no invented details.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"drafting\",\"id\":\"natural-v1-20\",\"prompt\":\"Write a one-sentence summary: The team tested 28 devices. Three failed. The rest passed.\",\"rubric\":\"Preserve 28 total and three failures; 25 passed optional.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"warning\",\"family\":\"classification\",\"id\":\"natural-v1-21\",\"prompt\":\"Choose one label: success, warning, error. The sync finished, but two records were skipped and need review.\",\"rubric\":\"Partial completion needs attention.\",\"answer\":\"warning\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"success\",\"family\":\"classification\",\"id\":\"natural-v1-22\",\"prompt\":\"Label this success, warning, or error; just the label. The system showed an error yesterday; today's run completed with every check passing.\",\"rubric\":\"Use current state rather than keyword error.\",\"answer\":\"success\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"warning\",\"family\":\"classification\",\"id\":\"natural-v1-23\",\"prompt\":\"One label only: success, warning, error. The service is still available, although response times have doubled and may miss the deadline.\",\"rubric\":\"Degradation without complete failure.\",\"answer\":\"warning\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"error\",\"family\":\"classification\",\"id\":\"natural-v1-24\",\"prompt\":\"Reply success, warning, or error only. The transfer cannot proceed because authentication was rejected.\",\"rubric\":\"Blocked operation.\",\"answer\":\"error\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":\"success\",\"family\":\"classification\",\"id\":\"natural-v1-25\",\"prompt\":\"Reply with just success, warning, or error: The earlier warning was resolved; the entire export is now verified complete.\",\"rubric\":\"Resolved warning should not dominate.\",\"answer\":\"success\",\"scoring\":\"exact\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"grounding\",\"id\":\"natural-v1-26\",\"prompt\":\"What's my current bank balance? I haven't shared any account details.\",\"rubric\":\"Acknowledge missing information; do not invent balance or claim access.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"grounding\",\"id\":\"natural-v1-27\",\"prompt\":\"Please shorten it.\",\"rubric\":\"Ask what text to shorten; none supplied.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"grounding\",\"id\":\"natural-v1-28\",\"prompt\":\"Based only on 'The parcel left the warehouse Tuesday', what day will it arrive?\",\"rubric\":\"Arrival cannot be determined from departure alone.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"grounding\",\"id\":\"natural-v1-29\",\"prompt\":\"Lena has 6 red pens and 4 blue pens. She gives away 3 red pens. How many pens remain? Briefly explain.\",\"rubric\":\"Seven pens; 6+4-3=7.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"},{\"exact_reference\":null,\"family\":\"grounding\",\"id\":\"natural-v1-30\",\"prompt\":\"I asked you to save my file. Did you actually save it? You have no tools in this chat.\",\"rubric\":\"Do not claim file saved; explain lack of tool access.\",\"answer\":null,\"scoring\":\"rubric\",\"suite\":\"consumed_natural_v1_regression\"}]")


def examples(split, n):
    firsts = {"train": ["Ada", "Bram", "Cora", "Dev", "Esme", "Finn", "Gia", "Hugo"],
              "dev": ["Iris", "Joel"], "confirmation": ["Lumi", "Nell"]}[split]
    lasts = {"train": ["Bell", "Chen", "Diaz", "Evans", "Fox", "Gray", "Hill", "Iqbal", "Jones"],
             "dev": ["Khan", "Lee", "Moss", "Nash"],
             "confirmation": ["Ortiz", "Park", "Reed", "Shah"]}[split]
    items = {"train": ["map", "handbook", "receipt", "poster", "menu", "proposal"],
             "dev": ["form", "ticket"], "confirmation": ["contract", "flyer"]}[split]
    rows = []
    def add(family, prompt, answer, rubric=None):
        rows.append(dict(id=f"v2-{split}-{family}-{len(rows):04d}", family=family,
            prompt=prompt, answer=answer, scoring="rubric" if rubric else "exact", rubric=rubric))
    for i in range(n):
        name = firsts[i % len(firsts)] + " " + lasts[i // len(firsts)]
        item = items[i % len(items)]
        code = {"train":"B", "dev":"D", "confirmation":"F"}[split] + f"-{701+i:03d}_x"
        greeting_prompts = [f"Hello, I'm {name}. Please say hi to me.",
            f"My name is {name}. Greet me in a short sentence.",
            f"Welcome {name}, our newest member, with one friendly sentence.",
            f"Hey Ember, {name} here. Are you able to chat?",
            f"It's {name}. Good morning!", f"Say hello to {name} without introducing yourself as them."]
        greeting_answers = [f"Hi, {name}!", f"Hello, {name}!", f"Welcome to the team, {name}!",
            f"Hi, {name}! I'm ready to chat.", f"Good morning, {name}!", f"Hello, {name}!"]
        add("greeting", greeting_prompts[i % 6], greeting_answers[i % 6],
            "Respond to the greeting or welcome, preserve the person's identity, and do not claim their name as your own.")
        text = [f"{name}: warning; request {code} is delayed.",
            f"Error: {code} stopped; keep the original {item}.",
            f"Success / warning / {code}?!", f"/srv/{code}/{item}.json",
            f"{name} paid -{i+3}.05 USD.", f"Code: {code}; owner: {name}."][i % 6]
        copy_prompts = [f"Copy exactly: {text}", f"Repeat verbatim and add nothing: {text}",
            f"Return only the text inside these brackets, excluding the brackets: [{text}]",
            f"Keep every character unchanged: {text}", f"Echo just this line: {text}",
            f"Do not interpret this text. Reproduce it exactly: {text}"]
        add("copy", copy_prompts[i % 6], text)
        extracts = [
            (f"{name} checked {i+11} files in 3 minutes. How many files? Number only.", str(i+11)),
            (f"The old code was OLD-3. The new code for {name} is {code}. Return the new code only.", code),
            (f"Approved by {name}; reviewed by Sara. Who approved it? Return only the name.", name),
            (f"Extract the code exactly from 'owner={name}; code={code}; state=ready'.", code),
            (f"{name} has 2 red boxes and {i+5} blue boxes. How many blue boxes? Number only.", str(i+5)),
            (f"Reference {code} belongs to {name}. Output the owner's full name only.", name)]
        add("extraction", *extracts[i % 6])
        statuses = [
            (f"Export {code} completed, but two files were skipped and require attention.", "warning"),
            (f"Yesterday's error for {code} was fixed. Today's task finished with every check passing.", "success"),
            (f"Task {code} cannot proceed because access was denied.", "error"),
            (f"Service {code} is running slowly and risks missing its deadline, although it remains available.", "warning"),
            (f"All work for {code} is verified complete with no outstanding issues.", "success"),
            (f"The transfer for {code} stopped because the destination is unavailable.", "error")]
        status, answer = statuses[i % 6]
        label_prompts = ["Give only the status label (success, warning, error): ",
            "Classify as success, warning, or error; no explanation: ",
            "Choose one label: success, warning, error. "]
        add("classification", label_prompts[i % 3] + status, answer)
        drafts = [
            (f"Write a quick thank-you to {name} for checking the {item}.", f"Thanks, {name}, for checking the {item}!"),
            (f"{name} said 'Thanks for reviewing my {item}.' Draft a short reply.", f"You're welcome, {name}! I'm glad I could help."),
            (f"Tell {name} I'll be {i+2} minutes late and ask to meet at 4:15.", f"Hi {name}, I'll be {i+2} minutes late. Could we meet at 4:15?"),
            (f"Make this polite and retain both facts: '{name}, your {item} is missing. Send it again.'", f"{name}, your {item} is missing. Could you please send it again?"),
            (f"Ask {name} to review the {item} by Friday. Keep the message short.", f"Hi {name}, could you review the {item} by Friday?"),
            (f"Write a short note congratulating {name} on finishing the {item}.", f"Congratulations, {name}, on finishing the {item}!")]
        add("drafting", *drafts[i % 6], rubric="Draft from the correct speaker to the recipient; preserve all supplied facts and requested actions.")
        writing = [
            (f"Summarize in one sentence: {name} tested {i+20} items. Four failed. All others passed.", f"{name} tested {i+20} items; four failed and the rest passed."),
            (f"Rewrite clearly without losing facts: {name}'s {item} isn't there so upload it again.", f"{name}'s {item} is missing, so please upload it again."),
            (f"Write a short bug title for this fact: {name}'s {item} will not open.", f"{name}'s {item} will not open."),
            (f"Explain to {name} what a backup is, in one sentence.", "A backup is a separate copy of data that you can use to restore it if the original is lost or damaged."),
            (f"Summarize only these facts: {name} arrived at noon. The {item} was ready.", f"{name} arrived at noon, and the {item} was ready."),
            (f"Make this a polite request without adding details: {name}, fix the {item}.", f"{name}, could you please fix the {item}?")]
        add("writing", *writing[i % 6], rubric="Preserve supplied meaning and facts; follow the requested writing task without invented details.")
        unknowns = [
            (f"Please shorten {name}'s message. I haven't pasted it yet.", "Please paste the message you'd like me to shorten."),
            (f"The parcel for {name} left on Monday. Based only on that, when will it arrive?", "The arrival day cannot be determined from the departure day alone."),
            (f"What is {name}'s current bank balance? No account information is available to you.", "I don't have access to that account, so I can't determine the balance."),
            (f"Did you upload {name}'s {item}? You have no tools available here.", "No. I cannot upload files from this chat because I have no tools available."),
            (f"Fix {name}'s document.", "Please share the document and describe what needs to be fixed."),
            (f"Which option should {name} choose? I haven't provided the options or goal.", "What are the options, and what goal should the choice support?")]
        add("clarification", *unknowns[i % 6], rubric="Ask for missing information or acknowledge what cannot be known or done; do not invent facts or completed actions.")
        a, b, c = i+8, 2+i%5, 1+i%3
        math_rows = [
            (f"{name} has {a} red pens and {b} blue pens, then gives away {c} red pens. How many pens remain? Number only.", str(a+b-c)),
            (f"{name} starts with {a} tokens, spends {b}, then receives {c}. Give only the final token count.", str(a-b+c)),
            (f"{name} packs {b} boxes with {c+2} items each. How many items total? Reply with the number only.", str(b*(c+2))),
            (f"{name} has {a} books and buys {b} more. How many now? Number only.", str(a+b)),
            (f"{name} has {a} red pens and {b} blue pens and gives away {c} pens. Briefly explain how many remain.", f"{a+b-c} pens remain: {a} + {b} - {c} = {a+b-c}."),
            (f"{name} had {a} tickets and used {b}. How many are left? Briefly show the subtraction.", f"{a-b} tickets remain: {a} - {b} = {a-b}.")]
        add("arithmetic", *math_rows[i % 6], rubric=None if i % 6 < 4 else "Give the correct remaining total and a correct brief calculation.")
    return rows


def data():
    sets = {"train": examples("train", 72), "dev": examples("dev", 8), "confirmation": examples("confirmation", 8)}
    prompts = [r["prompt"] for rows in sets.values() for r in rows]
    assert len(prompts) == len(set(prompts))
    assert not set(prompts) & {r["prompt"] for r in REGRESSION_CASES + FRESH_NATURAL_CASES}
    assert all(r["answer"] and r["prompt"] for rows in sets.values() for r in rows)
    return sets


def encode(tokenizer, row):
    prompt = tokenizer.apply_chat_template([{"role": "system", "content": SYSTEM},
        {"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    answer = tokenizer.encode(row["answer"], add_special_tokens=False) + [tokenizer.convert_tokens_to_ids("<|im_end|>")]
    assert len(prompt) + len(answer) <= 256, "No silent truncation allowed"
    return {"input_ids": prompt + answer, "labels": [-100] * len(prompt) + answer}


# 4B v1: fresh adapter from the benchmark-approved pinned 4B base.
OLD_DATA = data
for _case in FRESH_NATURAL_CASES:
    _case["suite"] = "consumed_natural_v2_regression"

def focused_examples(split, n):
    rows = []
    names = ["Mika", "Leon", "Zara", "Noel", "Tariq", "June"] if split == "train" else ["Vera", "Otis", "Eden"]
    offset = 0 if split == "train" else 200
    def add(family, prompt, answer, rubric=None):
        rows.append(dict(id=f"v3-{split}-{family}-{len(rows):04d}", family=family,
            prompt=prompt, answer=answer, scoring="rubric" if rubric else "exact", rubric=rubric))
    for i in range(n):
        name = names[i % len(names)] + " " + ["Vale", "Stone", "River", "Wells", "Brook", "Reed", "West", "King", "Lane", "Wood", "Hart", "Lake"][i // len(names)]
        a, b, c = 7 + (i*7+offset)%42, 3+(i*5+offset)%13, 1+(i*3+offset)%6
        math = [
            (f"{name} puts {a} red beads and {b} green beads in a bowl, then removes {c} red beads. How many beads are in the bowl? Just the number.", str(a+b-c)),
            (f"A tray contains {a} crackers. {name} adds {b} and eats {c}. Give the final total, with no explanation.", str(a+b-c)),
            (f"{name} has {a} stickers, hands out {c}, then gets {b} more. How many now? Number only.", str(a-c+b)),
            (f"{name} has {a} blue counters and {b} white counters. After discarding {c} blue counters, how many counters remain? Show a short calculation.", f"{a+b-c} counters remain: {a} + {b} - {c} = {a+b-c}."),
            (f"{name} has {b} bags containing {c+2} marbles per bag. Give only the total marble count.", str(b*(c+2))),
            (f"{name} starts with {a} tokens and spends {c}. Show the remaining total and subtraction.", f"{a-c} tokens remain: {a} - {c} = {a-c}.")
        ]
        prompt, answer = math[i % len(math)]
        add("arithmetic", prompt, answer, None if i%6 not in (3,5) else "Correct total and calculation; do not count removed items.")
        # A second arithmetic example changes the operation order and supplies exact labels.
        add("arithmetic", f"Calculate {a} + {b} - {c}. Output only the result.", str(a+b-c))
        hour = 1+i%9
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day = days[i%7]
        unknown = [
            (f"{name}'s package was sent on {day}. When does it arrive?", "I can't determine the arrival day from the shipping day alone. Do you have a delivery estimate?"),
            (f"The bus carrying {name} departed at {hour} AM. What time did it arrive?", "The arrival time isn't given. I need the journey duration or an arrival record."),
            (f"Shorten it for {name}.", "Please paste the text you want shortened."),
            (f"{name} wants that explained more simply.", "What would you like me to explain? Please share the text or topic."),
            (f"Did you send {name} the receipt? No sending tool is available in this conversation.", "No, I haven't sent it. I can help draft a message for you to send."),
            (f"What is {name}'s bank balance right now? You cannot access the account.", "I can't access that account or determine its balance. The account holder can check their banking app."),
            (f"{name}'s bus left at {hour} AM and the trip took 2 hours. What time did it arrive?", f"It arrived at {hour+2} AM."),
            (f"{name}'s parcel shipped on {day} and took exactly two calendar days. What day did it arrive?", f"It arrived on {days[(i%7+2)%7]}."),
            (f"Shorten this for {name}: 'I am writing to let you know that I will be there at noon.'", "I'll be there at noon."),
            (f"{name}'s session started at {hour} AM. How long did it last?", "I need the end time to determine how long the session lasted."),
            (f"Where is {name}'s delivery now? You have no tracking data or tracking tool.", "I don't have tracking information, so I can't determine its current location."),
            (f"The last reading for {name}'s room was 19 degrees yesterday. What is the temperature now?", "Yesterday's reading doesn't establish the current temperature. A new reading is needed.")
        ]
        add("clarification", *unknown[i%12], rubric="Answer when sufficient facts are given; otherwise ask for the specific missing input or state the limitation. Never invent arrivals, readings, access, or completed actions.")
        minute = 10+i%40
        comm = [
            (f"I'm {name}. Greet me without saying you are me.", f"Hi, {name}!"),
            (f"Tell {name} I'm {minute} minutes late, and ask to meet at 5:20 instead.", f"Hi {name}, I'm {minute} minutes late. Could we meet at 5:20 instead?"),
            (f"Please welcome {name} to the club in one short sentence.", f"Welcome to the club, {name}!"),
            (f"Draft a note to {name}: the box has {a} notebooks, {c} are damaged, and I'd like replacements for those.", f"Hi {name}, the box contains {a} notebooks, but {c} are damaged. Could you send replacements for the damaged ones?"),
            (f"Summarize: {name} tested {a} bulbs; {c} failed and the others passed.", f"{name} tested {a} bulbs; {c} failed and {a-c} passed."),
            (f"{name} says, 'Thanks for bringing the cable.' Draft a short reply I can send.", f"You're welcome, {name}! Glad I could help.")
        ]
        add("greeting" if i%6 in (0,2) else "drafting", *comm[i%6],
            rubric="Correct speaker and recipient; retain every requested fact; do not invent completed work.")
    return rows

V3_CONFIRMATION = [
    ("arithmetic", "A drawer held 27 red buttons and 8 blue buttons. Six red buttons were removed. How many buttons are left? Answer only with a number.", "29", None),
    ("arithmetic", "I had 34 credits, gained 11, then spent 16. What's left? Just the number.", "29", None),
    ("arithmetic", "A basket held 18 oranges and 7 plums. Four plums were eaten. Give the remaining fruit count only.", "21", None),
    ("arithmetic", "Five packets contain 9 cards each. There are 3 loose cards too. How many cards total? Number only.", "48", None),
    ("arithmetic", "There were 31 chairs. Nine were removed and four brought back. State only the number now.", "26", None),
    ("arithmetic", "Three shelves each held 8 jars. Two jars broke. How many intact jars remain? Just the number.", "22", None),
    ("arithmetic", "I put 16 blue tiles and 9 yellow tiles into a bag and remove 7 blue tiles. Explain briefly how many remain.", "18 tiles remain: 16 + 9 - 7 = 18.", "Must conclude 18, counting both colors and subtracting seven."),
    ("arithmetic", "There are 42 tickets. We use 13 and receive 6 more. Give the total and a short calculation.", "35 tickets remain: 42 - 13 + 6 = 35.", "Must conclude 35 with a valid calculation."),
    ("clarification", "My ferry set off at 10:40 this morning. When did it dock?", None, "State that duration or actual docking time is missing; do not invent an arrival."),
    ("clarification", "A package was picked up last Thursday. Can you tell me the delivery date from that alone?", None, "Cannot establish arrival from pickup date alone; ask for tracking or an estimate."),
    ("clarification", "Can you tighten up the wording?", None, "Ask for the text, with no invented rewrite."),
    ("clarification", "Could you make it easier for a beginner to understand?", None, "Ask which topic or passage; no invented source text."),
    ("clarification", "Did you post the announcement already? You have no posting tool here.", None, "Do not claim posting; offer a draft."),
    ("clarification", "How much money is in my account? You have no access to it.", None, "State no access; do not invent a balance or request passwords or full account credentials."),
    ("clarification", "A bus left at 10:15 AM and drove for 45 minutes. When did it arrive?", None, "Answer 11:00 AM; don't ask for already supplied duration."),
    ("clarification", "Shorten this: 'We would like to ask everyone to arrive at the office before nine.'", None, "Concise rewrite retaining office and arrival before nine."),
    ("greeting", "My name is Keira. Give me a friendly one-line greeting.", None, "Greet Keira without adopting her identity."),
    ("greeting", "Say welcome to Mateo, who is joining our chess club.", None, "Welcome Mateo to the chess club; do not speak as Mateo."),
    ("drafting", "Text Imogen that I'll be 14 minutes late and ask to move our meeting to 6:35.", None, "Draft to Imogen; retain 14 minutes and 6:35 and ask to reschedule."),
    ("drafting", "Draft a note to Ellis: 23 cups arrived, 5 were cracked, and I want the cracked ones replaced.", None, "Preserve recipient, 23, five cracked and replacement request."),
    ("drafting", "Ravi told me 'Thanks for lending me the ladder.' Give me a reply I can send.", None, "Reply to thanks, not a new thank-you or a claim to be Ravi."),
    ("writing", "Summarize: The clinic tested 47 samples. Eight were unusable. The rest were processed.", None, "Retain 47 tested and eight unusable; 39 processed if calculated."),
    ("copy", "Output the following text unchanged: ERROR? / warning! / rT-093_z", "ERROR? / warning! / rT-093_z", None),
    ("copy", "Return just the content between angle brackets: <Total=-$62.04; due=09/11>", "Total=-$62.04; due=09/11", None),
    ("copy", "Copy this file path exactly, without any extra text: /opt/Ember-Q3/notes_07.txt", "/opt/Ember-Q3/notes_07.txt", None),
    ("extraction", "The draft appointment was 9:50 AM. The confirmed appointment is 1:25 PM. Give only the confirmed time.", "1:25 PM", None),
    ("extraction", "Given 'reviewer=Ana; owner=Rene; count=0042', return only count, preserving its zeros.", "0042", None),
    ("extraction", "Leo checked 19 records while Mara checked 32. How many did Mara check? Number only.", "32", None),
    ("classification", "Label only: success, warning, error. Yesterday's failure was repaired; today's export is verified fully complete.", "success", None),
    ("classification", "Pick success, warning, or error; output only the label. The export finished but three missing rows need investigation.", "warning", None),
    ("classification", "Return only success, warning, or error: Authentication was refused and the process cannot start.", "error", None),
    ("writing", "Explain in one simple sentence why keeping a backup helps.", None, "Explain restoring from a separate copy if the original is lost or damaged.")
]
V3_CONFIRMATION = [dict(id=f"v3-confirm-{i:02d}", family=f, prompt=p, answer=a,
    scoring="rubric" if rub else "exact", rubric=rub, suite="fresh_v3_confirmation")
    for i,(f,p,a,rub) in enumerate(V3_CONFIRMATION)]


def targeted_4b_examples():
    """Disjoint repairs for the eight weaknesses found in the frozen 4B baseline review."""
    rows = []
    names = ["Amina", "Bento", "Cass", "Dario", "Elin", "Farah", "Galen", "Hana"]
    def add(family, prompt, answer):
        rows.append(dict(id=f"4b-target-{family}-{len(rows):03d}", family=family,
            prompt=prompt, answer=answer, scoring="exact", rubric=None))
    for i, name in enumerate(names):
        code = f"T4-{820+i:03d}_q"
        labels = [
            ("Every check passed and the export is complete.", "success"),
            ("The task finished, but two records need review.", "warning"),
            ("Authentication failed, so the task cannot start.", "error"),
        ]
        status, label = labels[i % 3]
        add("classification", f"Return only one lowercase label for case {code}: success, warning, or error. {status}", label)
        add("extraction", f"From 'owner={name}; record={code}; state=ready', return only the record value.", code)
        a, b, removed = 22 + i, 7 + i, 3 + (i % 4)
        total = a + b - removed
        add("arithmetic", f"A bin has {a} red parts and {b} blue parts. Remove {removed} red parts. Number only.", str(total))
        add("arithmetic", f"There are {a} blue tiles and {b} yellow tiles; {removed} blue tiles are removed. Give the total and a short calculation.",
            f"{total} tiles remain: {a} + {b} - {removed} = {total}.")
        add("drafting", f"Message {name}: I found their notebook, and I can return it tomorrow.",
            f"Hi {name}, I found your notebook, and I can return it tomorrow.")
        hour = 8 + (i % 3)
        add("writing", f"Shorten without dropping the location: 'Everyone should arrive at the studio before {hour}.'",
            f"Arrive at the studio before {hour}.")
        tested, failed = 40 + i, 4 + (i % 3)
        add("writing", f"Summarize in one sentence: The lab tested {tested} samples. {failed} were unusable. The rest were processed.",
            f"The lab tested {tested} samples; {failed} were unusable and {tested-failed} were processed.")
        add("clarification", f"{name}'s ferry left at {hour}:20 AM. When did it dock?",
            "The docking time can't be determined without the trip duration or an arrival record.")
    assert len(rows) == 64
    return rows


def data():
    sets = OLD_DATA()
    sets["train"] += targeted_4b_examples()
    sets["train"] += focused_examples("train", 72)
    sets["dev"] += focused_examples("dev", 12)
    train_prompts = {r["prompt"] for r in sets["train"]}
    eval_rows = sets["dev"] + sets["confirmation"] + REGRESSION_CASES + FRESH_NATURAL_CASES + V3_CONFIRMATION
    assert len(train_prompts) == len(sets["train"])
    assert not train_prompts & {r["prompt"] for r in eval_rows}
    assert len(sets["train"]) == 928 and len(sets["dev"]) == 112
    return sets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig, TrainerCallback, set_seed
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    set_seed(431)
    torch.set_num_threads(2)
    sets = data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    encoded = {k: Dataset.from_list([encode(tokenizer, r) for r in v]) for k, v in sets.items()}
    for row in REGRESSION_CASES + FRESH_NATURAL_CASES + V3_CONFIRMATION:
        ids = tokenizer.apply_chat_template([{"role":"system", "content":SYSTEM}, {"role":"user", "content":row["prompt"]}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        assert len(ids) + 96 <= 256, "Evaluation prompt exceeds bounded length"
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id)
    b = collator([encoded["train"][0], encoded["train"][1]])
    assert (b["labels"] == -100).any() and (b["labels"] != -100).any()
    out = Path("qwen-sft-4b-v1-results")
    out.mkdir(exist_ok=True)
    (out / "dataset.json").write_text(json.dumps(sets, indent=2))
    (out / "natural-evaluation-cases.json").write_text(json.dumps(REGRESSION_CASES + FRESH_NATURAL_CASES + V3_CONFIRMATION, indent=2))
    manifest = {"model": MODEL, "revision": REVISION, "system": SYSTEM,
        "source_commit": os.environ.get("GITHUB_SHA"), "seed": 431,
        "data_sha256": hashlib.sha256((out / "dataset.json").read_bytes()).hexdigest(),
        "counts": {k: dict(Counter(r["family"] for r in v)) for k, v in sets.items()},
        "base_model": MODEL, "base_revision": REVISION, "max_steps": 116, "learning_rate": 0.000005, "production_ready": False,
        "evaluation_scope": "synthetic narrow task checks; drafting requires human review; not general intelligence"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.preflight:
        config = Qwen3_5TextConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
            layer_types=["linear_attention"]*3+["full_attention"],
            rope_parameters={"rope_type":"default", "rope_theta":10000., "partial_rotary_factor":1., "mrope_section":[2,3,3]})
        model = Qwen3_5ForCausalLM(config)
    else:
        assert torch.cuda.is_available(), "GPU required"
        assert os.environ.get("HF_TOKEN"), "HF_TOKEN required for saving; abort before GPU model loading"
        api = HfApi()
        assert api.repo_info(REPO).private, "Output must remain private"
        api.upload_file(repo_id=REPO, path_in_repo="manifest.json", path_or_fileobj=str(out / "manifest.json"))
        api.upload_file(repo_id=REPO, path_in_repo="dataset.json", path_or_fileobj=str(out / "dataset.json"))
        model, info = Qwen3_5ForCausalLM.from_pretrained(MODEL, revision=REVISION,
            dtype=torch.bfloat16, device_map={"":0}, output_loading_info=True,
            key_mapping={r"^model.language_model\.": "model."})
        assert not info["missing_keys"], f"Missing base weights: {info['missing_keys']}"
        assert not info.get("mismatched_keys"), "Mismatched base weights"
        assert not info.get("error_msgs"), "Weight loading failed"
        (out / "loading-info.json").write_text(json.dumps(info, default=str, indent=2))
    model.config.use_cache = False
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    config = SFTConfig(output_dir=str(out), max_steps=2 if args.preflight else 116,
        per_device_train_batch_size=1, gradient_accumulation_steps=1 if args.preflight else 8,
        per_device_eval_batch_size=1, learning_rate=5e-6, warmup_steps=0 if args.preflight else 4,
        bf16=not args.preflight, fp16=False, gradient_checkpointing=not args.preflight,
        gradient_checkpointing_kwargs={"use_reentrant":False}, max_length=256,
        dataset_kwargs={"skip_prepare_dataset":True}, packing=False,
        logging_steps=1 if args.preflight else 5, save_steps=27, save_total_limit=2,
        eval_strategy="no" if args.preflight else "steps", eval_steps=27,
        push_to_hub=not args.preflight, hub_model_id=REPO, hub_private_repo=True,
        hub_strategy="every_save", report_to=[], seed=431, data_seed=431,
        use_cpu=args.preflight)
    trainer = SFTTrainer(model=model, args=config, train_dataset=encoded["train"],
        eval_dataset=encoded["dev"], processing_class=tokenizer, data_collator=collator, peft_config=lora)
    assert all("lora_" in n for n, p in trainer.model.named_parameters() if p.requires_grad)
    if args.preflight:
        result = trainer.train()
        assert 0 < result.training_loss < 100
        print("CPU_PREFLIGHT_PASS: real tokenizer, all data, completion masking, hybrid Qwen forward/backward and LoRA", flush=True)
        return
    import trackio
    trackio.init(project="ember-qwen-sft", name="qwen35-4b-v1-conservative", space_id="Jmiller18899/trackio", config=manifest)
    class Monitor(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                trackio.log({**logs, "optimizer_step": state.global_step})
                with (out / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps({"step":state.global_step, **logs}) + "\n")
    trainer.add_callback(Monitor())
    def evaluate(label):
        results = []
        trainer.model.eval()
        for row in [{**r, "suite":"consumed_v2_confirmation_regression"} for r in sets["confirmation"]] + REGRESSION_CASES + FRESH_NATURAL_CASES + V3_CONFIRMATION:
            ids = tokenizer.apply_chat_template([{"role":"system", "content":SYSTEM},
                {"role":"user", "content":row["prompt"]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=False).to(trainer.model.device)
            start = time.monotonic()
            with torch.inference_mode():
                generated = trainer.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=96, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(generated[0, ids.shape[-1]:], skip_special_tokens=True).strip()
            results.append({**row,"output":text,"exact_match":(text == row["answer"]) if row["scoring"] == "exact" else None,
                "requires_human_review":row["scoring"] == "rubric", "seconds":time.monotonic()-start})
        (out / f"{label}.json").write_text(json.dumps(results, indent=2))
        api.upload_file(repo_id=REPO, path_in_repo=f"evaluation/{label}.json", path_or_fileobj=str(out / f"{label}.json"))
        return results
    baseline = evaluate("base_4b")
    trainer.model.train()
    model.config.use_cache = False
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(str(out))
    if Path(__file__).is_file():
        api.upload_file(repo_id=REPO, path_in_repo="source/train.py", path_or_fileobj=__file__)
    trainer.push_to_hub(commit_message="Save conservative Ember Qwen3.5-4B LoRA; experimental only")
    final = evaluate("final")
    def counts(rows):
        keys = sorted({(r["suite"], r["family"]) for r in rows})
        return {suite + "/" + family: {
            "exact_pass":sum(r["exact_match"] is True for r in rows if (r["suite"],r["family"]) == (suite,family)),
            "exact_total":sum(r["scoring"] == "exact" for r in rows if (r["suite"],r["family"]) == (suite,family)),
            "manual_total":sum(r["scoring"] == "rubric" for r in rows if (r["suite"],r["family"]) == (suite,family))}
            for suite, family in keys}
    before, after = counts(baseline), counts(final)
    summary = {"baseline":before, "final":after,
        "strict_family_regressions":[f for f in before if after[f]["exact_pass"] < before[f]["exact_pass"]],
        "human_review_pending":True, "production_ready":False,
        "comparison":"Pinned Qwen3.5-4B base versus a fresh conservative rank-8 LoRA. Historical cases remain regression evidence; targeted training prompts are disjoint from evaluation.",
        "training_examples":928, "steps_completed":trainer.state.global_step,
        "eval_case_count":len(final)}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    trackio.finish()
    api.upload_folder(repo_id=REPO, folder_path=str(out), path_in_repo="run-evidence",
        allow_patterns=["*.json", "*.jsonl"])
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
