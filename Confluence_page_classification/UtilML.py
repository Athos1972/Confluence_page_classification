import textstat
import torch
from yake import KeywordExtractor


class UtilML:
    @staticmethod
    def get_stopwords_de() -> list:
        """
        Get a list of German stopwords
        :return: List of stopwords
        """
        return [
            # Eine umfangreiche Liste deutscher Stopwörter hinzufügen
            "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am", "an", "ander", "andere", "anderem",
            "anderen",
            "anderer", "anderes", "anderm", "andern", "anderr", "anders", "auch", "auf", "aus", "bei", "bin", "bis",
            "bist",
            "da", "damit",
            "dann", "der", "den", "des", "dem", "die", "das", "daß", "derselbe", "derselben", "denselben", "desselben",
            "demselben",
            "dieselbe", "dieselben", "dasselbe", "dazu", "dein", "deine", "deinem", "deinen", "deiner", "deines",
            "denn",
            "derer", "dessen",
            "dich", "dir", "du", "dies", "diese", "diesem", "diesen", "dieser", "dieses", "doch", "dort", "durch",
            "ein",
            "eine", "einem",
            "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges", "einmal", "er",
            "ihn",
            "ihm", "es",
            "etwas", "euer", "eure", "eurem", "euren", "eurer", "eures", "für", "gegen", "gewesen", "hab", "habe",
            "haben",
            "hat", "hatte",
            "hatten", "hier", "hin", "hinter", "ich", "mich", "mir", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres",
            "euch",
            "im", "in",
            "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem", "jenen", "jener",
            "jenes",
            "jetzt", "kann",
            "kein", "keine", "keinem", "keinen", "keiner", "keines", "können", "könnte", "machen", "man", "manche",
            "manchem",
            "manchen",
            "mancher", "manches", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mit", "muss", "musste",
            "nach",
            "nicht", "nichts",
            "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine", "seinem", "seinen", "seiner", "seines",
            "selbst", "sich",
            "sie", "ihnen", "sind", "so", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollte",
            "sondern",
            "sonst", "über",
            "um", "und", "uns", "unser", "unsere", "unserem", "unseren", "unserer", "unseres", "unter", "viel", "vom",
            "von",
            "vor", "während",
            "war", "waren", "warst", "was", "weg", "weil", "weiter", "welche", "welchem", "welchen", "welcher",
            "welches",
            "wenn", "werde",
            "werden", "wie", "wieder", "will", "wir", "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden",
            "zu", "zum",
            "zur", "zwar",
            "zwischen"]

    @staticmethod
    def flesch_kincaid(text):
        return textstat.flesch_reading_ease(text)

    @staticmethod
    def extract_keyphrases(text):
        kw_extractor = KeywordExtractor(lan="de", n=2, top=5)
        keywords = kw_extractor.extract_keywords(text)
        return [kw[0] for kw in keywords]

    @staticmethod
    def get_bert_embeddings(text, tokenizer, model):
        inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return torch.mean(last_hidden_states, dim=1).detach().numpy()

    @staticmethod
    def truncate_text(text, tokenizer, max_length=512):
        inputs = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
        return tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
