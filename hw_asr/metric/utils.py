# Don't forget to support cases when target_text == ''
import editdistance


def calc_cer(target_text, predicted_text) -> float:
    distance = editdistance.distance(target_text.split(), predicted_text.split())
    return distance / len(target_text.split()) if len(target_text.split()) != 0 else distance


def calc_wer(target_text, predicted_text) -> float:
    distance = editdistance.distance(target_text, predicted_text)
    return distance / len(target_text) if len(target_text) != 0 else distance
