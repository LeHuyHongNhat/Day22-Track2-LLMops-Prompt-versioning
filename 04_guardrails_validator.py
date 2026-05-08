"""
Step 4 — Guardrails AI Validators
====================================
Build two custom validators:
  A) PIIDetector — detects & redacts emails, phones, SSNs, credit cards
  B) JSONFormatter — validates and auto-repairs malformed JSON
"""

import re
import json
from guardrails import Guard
from guardrails.validators import (
    Validator,
    register_validator,
    PassResult,
    FailResult,
)
from guardrails.validator_base import OnFailAction


# ---- Validator A: PII Detector -----------------------------------------------
@register_validator(name="pii-detector", data_type="string")
class PIIDetector(Validator):
    """Detects and redacts Personally Identifiable Information (PII).

    Patterns: EMAIL, PHONE (US), SSN, CREDIT_CARD
    Uses PassResult with value_override to redact matched PII.
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Redacted {len(found_pii)} PII items: {[p[0] for p in found_pii]}")
            return FailResult(
                error_message=f"PII detected: {[p[0] for p in found_pii]}",
                fix_value=redacted_text,
            )
        return PassResult()


# ---- Validator B: JSON Formatter ---------------------------------------------
@register_validator(name="json-formatter", data_type="string")
class JSONFormatter(Validator):
    """Validates and auto-repairs malformed JSON strings.

    Repairs: strip markdown fences, fix single quotes, remove trailing commas.
    """

    @staticmethod
    def _repair(text: str) -> str:
        text = text.strip()
        # Remove markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        # Single quotes to double quotes
        text = text.replace("'", '"')
        # Remove trailing commas before } or ]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return text

    def validate(self, value: str, metadata: dict):
        # Try parsing as-is — valid JSON passes through
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        # Try repair then parse
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            repaired = json.dumps(parsed, indent=2)
            print("  🔧 JSON repaired successfully")
            return FailResult(
                error_message="JSON was malformed but repaired",
                fix_value=repaired,
            )
        except json.JSONDecodeError as e:
            print(f"  ❌ JSON repair failed: {e}")
            return FailResult(
                error_message=f"Invalid JSON after repair: {e}",
                fix_value=json.dumps({"error": "Invalid JSON", "raw": value[:200]}),
            )


# ---- PII Guard demo ----------------------------------------------------------
def demo_pii_guard():
    """Create a Guard with PIIDetector and test on 6 sample texts."""
    print("\n" + "=" * 55)
    print("  PII Detection Demo")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",       "Contact John at john.doe@example.com for details."),
        ("Phone",       "Call our support line at (555) 867-5309."),
        ("SSN",         "Patient SSN is 123-45-6789 on file."),
        ("Credit Card", "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",   "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",       "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ---- JSON Guard demo ---------------------------------------------------------
def demo_json_guard():
    """Create a Guard with JSONFormatter and test on 5 sample strings."""
    print("\n" + "=" * 55)
    print("  JSON Formatting Demo")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",        '{"name": "Alice", "age": 30}'),
        ("Markdown fences",   '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",     "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",    '{"key": "value",}'),
        ("Truly invalid",     "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text[:80]}")
        output = str(result.validated_output)
        print(f"  Output: {output[:80]}")


# ---- Main --------------------------------------------------------------------
def main():
    print("=" * 55)
    print("  Step 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Step 4 complete!")


if __name__ == "__main__":
    main()
