import re
import html

def preprocess_sms(text: str) -> str:
    """
    Advanced NLP Preprocessing for SMS Spam Detection (V2).
    Preserves crucial spam signals (URLs, Phones, Money, Urgency) via token replacement.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text).strip()

    url_pattern = r'(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.(?:com|co\.uk|org|net|me|info|biz|cc|tv)/\S*)'
    text = re.sub(url_pattern, ' tok_url ', text, flags=re.IGNORECASE)

    money_pattern = r'([£$€¥₹]\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:pounds?|dollars?|euros?|p|quid|cash|prize)\b)'
    text = re.sub(money_pattern, ' tok_money ', text, flags=re.IGNORECASE)

    phone_pattern = r'(\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}|\b\d{5,6}\b)'
    text = re.sub(phone_pattern, ' tok_phone ', text)

    text = re.sub(r'!{2,}', ' tok_multiexclam ', text)
    text = re.sub(r'!', ' tok_exclam ', text)
    text = re.sub(r'\?{2,}', ' tok_multiquestion ', text)

    text = re.sub(r'\b\d+\b', ' tok_num ', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text
