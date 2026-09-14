"""Additional development contrasts for the remaining v4 routing errors."""
from __future__ import annotations

from .routing_data_v4 import LABELS, load_training as load_v4

REVISION = "routing-development-v5"
FRAMES = {
    "weather": (
        "Could you find out whether it is wet outside in {place} right now?",
        "Tell me if the weather feels warm in {place} at this moment.",
        "Are we getting drizzle in {place} currently?",
        "Check for rain showers in {place} now.",
        "I am heading outside; is it damp in {place} at the moment?",
        "Is the air chilly in {place} right now?",
        "How muggy is it in {place} today?",
        "I am packing a coat; please check the current temperature in {place}.",
    ),
    "web_search": (
        "Look up ongoing train delays at {place} station.",
        "Search for current ferry service interruptions near {place}.",
        "Check the latest bus cancellations affecting {place}.",
        "Find official rail disruption notices for {place}.",
        "Look online for today's ferry timetable from {place}.",
        "Find the currently published tram schedule for {place}.",
        "Are any services suspended at {place} station? Verify online.",
        "Get current transport service notices for {place}.",
    ),
    "direct": (
        "Explain the meaning of {topic} without a current lookup.",
        "Write a fictional story that mentions {topic}.",
        "Make this phrase easier to understand: {topic}.",
        "Teach me the basic concept of {topic}.",
        "Give a made-up example illustrating {topic}.",
        "Translate these words into French: {topic}.",
        "Write a brief poem about {topic}.",
        "Suggest a heading for an article explaining {topic}.",
    ),
    "get_time": (
        "I am about to phone someone in {place}; what time is it there now?",
        "Tell me the current local time for {place}.",
        "I am calling a colleague in {place}; is it morning there right now?",
        "Check the clock in {place} right now, please.",
        "What hour is it locally in {place} now?",
        "I am planning to contact someone in {place}; what is the local time there?",
        "Show me the time in {place} at the moment.",
        "What time of day is it in {place} currently?",
    ),
    "calculator": (
        "Compute ({a}+{b})*3.", "Calculate {a} minus -{b}.",
        "Find {b} percent of {a}.", "What is {a} times ({b}+2)?",
        "Evaluate ({a}-{b})/2.", "Compute {a}.5 plus {b}.25.",
        "Calculate ({b}) squared.", "What is -({a}+{b})?",
    ),
}


def development_cases():
    places = ("Cuenca", "Tromsø", "Mombasa", "Salzburg")
    topics = ("railway service disruptions", "wet weather", "time zones", "a ferry timetable")
    return [{"id": f"v5-{frame:02d}-{label}-{variant}",
             "group": f"contrast-{frame:02d}", "route": label,
             "user": FRAMES[label][frame].format(place=places[variant], topic=topics[variant],
                        a=(37, 53, 71, 97)[variant], b=(6, 14, 22, 31)[variant])}
            for frame in range(8) for label in LABELS for variant in range(4)]


def load_training(root):
    historical, prior_development = load_v4(root)
    development = prior_development + development_cases()
    texts = [" ".join(c["user"].casefold().split()) for c in historical + development]
    if len(texts) != len(set(texts)):
        raise ValueError("Duplicate v5 training requests")
    return historical, development
