import csv
import time
from pathlib import Path

from rag_engine import get_rag_response

QUESTIONS = [
    "When do exams start?",
    "What month are semester 1 exams?",
    "Where can I see the exam timetable?",
    "What happens if I miss my exam?",
    "I was sick during my exam, what do I do?",
    "When are resit exams held?",
    "What do I need to bring to the exam?",
    "Can I sit my exam without my student ID?",
    "When is the exam schedule published?",
    "How do I apply for a resit?",
    "How do I sign up for my courses?",
    "What's the last day to register?",
    "Can I register late?",
    "How much is the late registration fee?",
    "Can I change my modules after registering?",
    "How do I add a new module?",
    "Where do I get the module change form?",
    "When does registration open?",
    "Is registration done online?",
    "Who approves late registration?",
    "How much do I need to pay?",
    "What are the tuition fees for undergraduates?",
    "How much is the postgraduate fee?",
    "Can I pay my fees in instalments?",
    "How many instalments are allowed?",
    "When is the first instalment due?",
    "What happens if I pay late?",
    "How much is the late payment penalty?",
    "Where do I pay my fees?",
    "Will I get my results if I owe fees?",
    "What time does the library open?",
    "What time does the library shut?",
    "Is the library open on Saturday?",
    "How many books can I borrow?",
    "How long can I keep a borrowed book?",
    "Can I renew my books?",
    "What's the fine for late book returns?",
    "Does the library have computers?",
    "Are there study rooms in the library?",
    "Where is the library located?",
    "Where do I find my class schedule?",
    "When are timetables published?",
    "What time do classes begin in the morning?",
    "What time is the last class?",
    "Are there evening classes?",
    "What time are part-time classes?",
    "How will I know about timetable changes?",
    "Where are room changes announced?",
    "When does the semester start?",
    "My class room changed, where do I check?",
    "How do I connect to the internet on campus?",
    "Is there free Wi-Fi for students?",
    "What's the Wi-Fi network name?",
    "I forgot my portal password, what do I do?",
    "How do I reset my password?",
    "Where is the IT helpdesk?",
    "What time is the IT helpdesk open?",
    "Do students get a university email?",
    "How do I access the online learning platform?",
    "What's the IT helpdesk email?",
    "I'm feeling stressed, who can I talk to?",
    "Does the university have a counsellor?",
    "Is counselling free?",
    "Is counselling confidential?",
    "How do I book a counselling appointment?",
    "Where is the wellbeing centre?",
    "What time is the wellbeing centre open?",
    "Is there a nurse on campus?",
    "What health services are available?",
    "How do I email the wellbeing centre?",
    "How do I apply for hostel?",
    "Does the university have dorms?",
    "How much does student housing cost?",
    "What's the cheapest hostel rate?",
    "How do I find approved accommodation?",
    "Who has the list of approved hostels?",
    "Can first-year students get accommodation?",
    "When should I apply for housing?",
    "How much is rent per month?",
    "Are hostels near the campus?",
    "When is the graduation ceremony?",
    "What do I need to do to graduate?",
    "Can I graduate if I owe fees?",
    "How do I register for graduation?",
    "When does graduation registration open?",
    "How do I get my certificate?",
    "What if I can't attend the ceremony?",
    "Do I need to pass all modules to graduate?",
    "Is graduation once a year?",
    "Where do I collect my certificate?",
    "How do I reach the admin office?",
    "What's the university's phone number?",
    "What's the registry email?",
    "What time is the registry open?",
    "Who do I email about my results?",
    "What's the finance office email?",
    "What's the university's address?",
    "What's the wellbeing centre email?",
    "How do I contact the finance office?",
    "Where is the university located?",
]

OUTPUT_FILE = Path("chatbot_test_results.csv")


def main():
    results = []
    total_start = time.perf_counter()

    for number, question in enumerate(QUESTIONS, start=1):
        start = time.perf_counter()

        try:
            answer = get_rag_response(question)
            error = ""
        except Exception as exc:
            answer = ""
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - start

        results.append({
            "No": number,
            "Question": question,
            "Chatbot Answer": answer,
            "Response Time (s)": round(elapsed, 3),
            "Error": error,
            "Status": "",
            "Notes": "",
        })

        print(f"[{number:03d}/100] {elapsed:.3f} s - {question}")

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "No",
                "Question",
                "Chatbot Answer",
                "Response Time (s)",
                "Error",
                "Status",
                "Notes",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    times = [row["Response Time (s)"] for row in results]
    total_elapsed = time.perf_counter() - total_start

    print("\nTesting completed.")
    print(f"Questions tested: {len(results)}")
    print(f"Average response time: {sum(times) / len(times):.3f} s")
    print(f"Fastest response: {min(times):.3f} s")
    print(f"Slowest response: {max(times):.3f} s")
    print(f"Total test time: {total_elapsed:.2f} s")
    print(f"CSV saved as: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
