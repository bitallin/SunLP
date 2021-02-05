from sunlp.phrase.ngram_model import NgramModel
from sunlp.preprocess import TextSplitter


class Phrase:
    def __init__(self):
        self.text_splitter = TextSplitter()

    def discover(self,
                 texts,
                 top_k: float = 200,
                 min_gram: int = 2,
                 max_gram: int = 6,
                 min_freq: int = 1,
                 chunk_size: int = 100000,
                 with_score: bool = False,
                 min_score: float = 0.0
                 ):

        """
            新词发现算法入口
        Args:
            texts: list[str]
            top_k:  取前k个new words或前k%的new words
            min_gram: min n-gram
            max_gram: max n-gram
            min_freq: 词最小的频率
            chunk_size: 批处理统计词频的chunk size
            with_score: T
            min_score: 可根据需求对成词的最低分数进行限制
        Returns:
            list: [ word1, word2 ....]  降序
        """
        sentences = (sent for text in texts for sent in self.text_splitter.split_text(text) if len(sent) > 2)

        # 基于ngram计算自由度, 凝聚度
        word_info_scores = NgramModel.get_scores(sentences, min_gram, max_gram, chunk_size, min_freq)

        new_words = []
        for item in sorted(word_info_scores.items(), key=lambda item_: item_[1][-1], reverse=True):
            if item[1][-1] <= min_score:
                continue
            new_words.append((item[0], item[1][-1]) if with_score else item[0])

        return new_words[:top_k] if top_k > 1 else new_words[:int(top_k * len(new_words))]
