"""Development-only intent contrasts; no confirmation requests are generated here."""
from __future__ import annotations

import json
from pathlib import Path

REVISION = "routing-development-v4"
LABELS = ("direct", "weather", "calculator", "web_search", "get_time")

# A frame group stays together across all tools, locations and numeric variants
# during cross-validation. Replacing a city is not an independent test example.
FRAMES = {
    "direct": [
        "Explain what {topic} means without looking anything up.",
        "Write a short fictional scene about {topic}.",
        "Give a beginner-friendly definition of {topic}.",
        "Rephrase this heading: {topic} needs a clearer explanation.",
        "Describe the basic idea behind {topic}; no current data is needed.",
        "Create a metaphor to help me understand {topic}.",
        "Suggest a title for an essay about {topic}.",
        "Draft a polite request for somebody to explain {topic}.",
        "Teach a child the meaning of {topic}.",
        "Turn these words into a complete sentence: {topic} is useful.",
        "Write a poem mentioning {topic}.",
        "Summarize this sentence: {topic} helps people organize information.",
        "Make a study plan for learning the concept of {topic}.",
        "Explain why people use {topic} in everyday life.",
        "Translate the phrase '{topic}' into Spanish.",
        "Write an example question about {topic}; do not answer that question.",
    ],
    "weather": [
        "What is the weather in {place} right now?",
        "Tell me how hot it is in {place} at the moment.",
        "Is rain falling in {place} now?",
        "Show the current conditions for {place}.",
        "How cold is it outside in {place} today?",
        "Check the temperature in {place}, please.",
        "Do I need an umbrella in {place} right now?",
        "Is it windy in {place} at this moment?",
        "Give me a weather update for {place}.",
        "How does the sky look in {place} now?",
        "Are the conditions in {place} sunny or cloudy currently?",
        "What's the current humidity in {place}?",
        "Has it started snowing in {place} right now?",
        "Fetch the latest weather observation for {place}.",
        "What would a thermometer outside in {place} read now?",
        "Is the weather dry in {place} at the moment?",
    ],
    "calculator": [
        "Calculate {a} + {b}.",
        "What is {a} times {b}?",
        "Work out ({a}+{b})/2.",
        "Evaluate -{a} plus {b}.",
        "Find {a} percent of {b}.",
        "Compute {a} squared.",
        "What do I get if I subtract {b} from {a}?",
        "Multiply ({a}-{b}) by 3.",
        "I need the value of {a}/{b}.",
        "How much is {a} minus {b}?",
        "{a} * ({b} + 4)",
        "Please solve {a}.5 + {b}.25.",
        "Give the result of ({a}+{b})**2.",
        "Do the arithmetic: {a} divided by {b}.",
        "Add {a} and {b} together.",
        "Can you calculate -({a}-{b})*2?",
    ],
    "web_search": [
        "Find the current application deadline for {subject}.",
        "Look up the latest service disruption notices for {subject}.",
        "What timetable is in effect today for {subject}?",
        "Search for the current admission prices for {subject}.",
        "Are there any newly announced changes to {subject}?",
        "Check whether {subject} is open to visitors today.",
        "Find the newest official announcement about {subject}.",
        "What are the latest news reports about {subject}?",
        "Look online for the latest booking availability for {subject}.",
        "Has registration opened yet for {subject}? Check the official site.",
        "Please find the current cancellation policy for {subject}.",
        "I need a link to the latest schedule for {subject}.",
        "Check the web for an update on {subject}.",
        "When is the next scheduled event for {subject}? Verify it online.",
        "Find recent public reports concerning {subject}.",
        "Verify the current contact information for {subject}.",
    ],
    "get_time": [
        "What time is it in {place} right now?",
        "Tell me the local time in {place}.",
        "Show the current clock time for {place}.",
        "What would a clock in {place} show at this moment?",
        "Is it morning or evening in {place} now?",
        "Check the time in {place}, please.",
        "Give me the current local hour in {place}.",
        "What is the time of day in {place} at the moment?",
        "I need to know what the clock says in {place} now.",
        "How late is it in {place} right now?",
        "Please get the current time for {place}.",
        "Can you tell me the time in {place} today?",
        "What's the hour in {place} currently?",
        "Read the local clock for {place}.",
        "Find out whether it is still afternoon in {place} now.",
        "Could you check what time it is in {place} at this instant?",
    ],
}


def development_cases():
    places = ("Malmö", "Tunis", "Anchorage", "Wellington")
    topics = ("a weather forecast", "a time zone", "a search engine", "a calculator")
    subjects = ("the regional ferry service", "the city art museum",
                "the university summer program", "the community science festival")
    rows = []
    for frame in range(16):
        for label in LABELS:
            for variation in range(4):
                rows.append({
                    "id": f"v4-{frame:02d}-{label}-{variation}",
                    "group": f"frame-{frame:02d}",
                    "route": label,
                    "user": FRAMES[label][frame].format(
                        place=places[variation], topic=topics[variation],
                        subject=subjects[variation], a=(23, 41, 67, 89)[variation],
                        b=(7, 11, 13, 17)[variation]),
                })
    return rows


def load_training(root: Path):
    original = json.loads((root / "tool_assistant/data/router-training.json").read_text())
    historical = [{"id": c["id"], "user": c["user"],
                   "route": "direct" if c["kind"] == "direct_response" else c["expected_tool"],
                   "group": "historical-training"} for c in original["cases"]]
    development = development_cases()
    # These cases are deliberately labeled development. The consumed v2
    # confirmation and parser v3 confirmation are excluded from fitting.
    seen = set()
    for case in historical + development:
        text = " ".join(case["user"].casefold().split())
        if text in seen:
            raise ValueError(f"Duplicate routing training request: {case['id']}")
        seen.add(text)
    return historical, development
