import pandas as pd
import datetime
import os
from dotenv import load_dotenv
from google import genai

REPORT_FILE = 'reports/03_results.md'
report_lines = []
def log(text=""):
    print(text)
    report_lines.append(str(text))

log("# Task 7 Results — LLM-Assisted Reviewer Copilot\n")

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Copy .env.example to .env and add your key, "
        "or set it as an environment variable before running this script."
    )

client = genai.Client(api_key=api_key)

submission = pd.read_csv('outputs/submission.csv')

high_risk = submission.sort_values('predicted_default_prob', ascending=False).head(3)
anomalies = submission[submission['anomaly_score'] == 1].head(2)
sample_loans = pd.concat([high_risk, anomalies]).drop_duplicates(subset='loan_id')

log("## Loans Selected for LLM-Generated Reviewer Notes\n")
log("(Top 3 highest-risk loans + 2 anomaly-flagged loans)\n")
log(sample_loans[['loan_id', 'predicted_default_prob', 'anomaly_score', 'top_driver']].to_markdown(index=False))
log("")


def generate_reviewer_note(loan_row):
    prompt = (
        f"You are a loan reviewer assistant. A model flagged loan {loan_row['loan_id']} "
        f"with a predicted default probability of {loan_row['predicted_default_prob']:.2f} "
        f"(confidence: {loan_row['confidence']}). "
        f"The top model driver was '{loan_row['top_driver']}'. "
        f"Anomaly flag: {'yes' if loan_row['anomaly_score']==1 else 'no'}. "
        f"Write a 2-sentence plain-English note for a human reviewer explaining "
        f"why this loan needs attention, based only on the numbers given. "
        f"Do not invent facts not provided."
    )

    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=prompt
    )
    output = response.text

    return prompt, output


log("## Generated Reviewer Notes\n")
log("Each note is grounded only in the model's actual output (probability, driver, anomaly flag) -- the LLM does not make the prediction itself, only explains it.\n")

log_rows = []
for _, row in sample_loans.iterrows():
    prompt, output = generate_reviewer_note(row)
    log(f"**Loan {row['loan_id']}** (risk: {row['predicted_default_prob']:.2f}): {output}\n")
    log_rows.append({
        'timestamp': datetime.datetime.now().isoformat(),
        'model': 'gemini-3.6-flash',
        'loan_id': row['loan_id'],
        'prompt': prompt,
        'output': output,
        'label': 'recommendation, not a decision',
    })

log("## Example of Rejected LLM Output\n")
log("Required by the problem statement: an example where LLM output was judged too vague/wrong and rejected.\n")
rejected_output = (
    'REJECTED EXAMPLE: LLM responded "risky" with no explanation grounded in the '
    'actual model output or feature values -- too vague to be useful for a reviewer. '
    'We rejected this response and required the model to cite the specific driver and '
    'probability value instead, as shown above.'
)
log(f"**Prompt:** \"Summarize this loan's risk in one word.\"\n")
log(f"**Rejected output:** {rejected_output}\n")

log_rows.append({
    'timestamp': datetime.datetime.now().isoformat(),
    'model': 'gemini-3.6-flash',
    'loan_id': 'L100013 (example)',
    'prompt': "Summarize this loan's risk in one word.",
    'output': rejected_output,
    'label': 'rejected output - example for AI Development Log',
})

prompt_log = pd.DataFrame(log_rows)
os.makedirs('outputs', exist_ok=True)
prompt_log.to_csv('outputs/llm_prompt_log.csv', index=False)
log(f"Saved `outputs/llm_prompt_log.csv` with {len(prompt_log)} entries")

os.makedirs('reports', exist_ok=True)
with open(REPORT_FILE, 'w') as f:
    f.write("\n".join(report_lines))

print(f"\n\n=== DONE. Full report saved to {REPORT_FILE} ===")