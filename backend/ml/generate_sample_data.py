"""
Generates a synthetic-but-realistic labeled dataset of log/ticket/alert
lines to train the demo classifier. Replace this with the client's real
historical log data before going to production — the training script
downstream doesn't care where the CSV came from, as long as it has
"text" and "label" columns.
"""
import csv
import random
from pathlib import Path

from backend import config

random.seed(42)

TEMPLATES = {
    "INFO": [
        "User {u} logged in successfully from {ip}",
        "Scheduled backup completed for database {db}",
        "Cache warmed up successfully in {ms}ms",
        "Health check passed for service {svc}",
        "Deployment of {svc} v{ver} completed successfully",
        "User {u} updated profile settings",
        "Daily report generated and emailed to {u}",
        "New user {u} registered successfully",
    ],
    "WARNING": [
        "API response time for {svc} exceeded 2s threshold ({ms}ms)",
        "Deprecated endpoint /{svc}/v1 called by client {ip}",
        "Retry attempt {n} for job {svc} after transient failure",
        "Disk usage on {svc} reached 78% capacity",
        "Rate limit approaching for API key ending in {n}",
        "Connection pool for {db} nearing max capacity",
        "SSL certificate for {svc} expires in 14 days",
    ],
    "ERROR": [
        "Failed to connect to database {db} after {n} retries",
        "NullPointerException in module {svc} at line {n}",
        "Payment processing failed for order #{n} - gateway timeout",
        "Unhandled exception in {svc}: TimeoutError",
        "Failed to send email to {u} - SMTP connection refused",
        "Request to {svc} failed with status 500",
        "Job {svc} failed after {n} minutes with exit code 1",
    ],
    "CRITICAL": [
        "Production database {db} is DOWN - all connections refused",
        "Service {svc} crashed - out of memory, restarting pod",
        "Payment gateway completely unresponsive for 5 minutes",
        "Full disk outage on {svc} - service unavailable",
        "Kubernetes node {svc} not responding - all pods evicted",
        "Fatal error: {svc} process terminated unexpectedly",
    ],
    "SECURITY_ALERT": [
        "Multiple failed login attempts detected for user {u} from {ip}",
        "SQL injection attempt blocked on endpoint /{svc}",
        "Unauthorized access attempt to admin panel from {ip}",
        "Suspicious API key usage pattern detected for key ending {n}",
        "Possible brute-force attack detected on login endpoint from {ip}",
        "Unusual data exfiltration pattern detected from {svc}",
    ],
}

USERS = ["priya", "rahul", "amit", "sneha", "vikram", "anjali"]
SERVICES = ["auth-service", "payments-api", "user-db", "checkout", "search-svc", "notif-worker"]
IPS = [f"192.168.{random.randint(1,254)}.{random.randint(1,254)}" for _ in range(30)]


def _fill(template: str) -> str:
    return template.format(
        u=random.choice(USERS),
        ip=random.choice(IPS),
        db=random.choice(SERVICES),
        svc=random.choice(SERVICES),
        ms=random.randint(100, 5000),
        n=random.randint(1, 999),
        ver=f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}",
    )


def generate(samples_per_label: int = 60) -> None:
    config.TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, templates in TEMPLATES.items():
        for _ in range(samples_per_label):
            rows.append((_fill(random.choice(templates)), label))
    random.shuffle(rows)

    with open(config.TRAIN_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    print(f"Generated {len(rows)} sample log entries -> {config.TRAIN_DATA_PATH}")


if __name__ == "__main__":
    generate()
