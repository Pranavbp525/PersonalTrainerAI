import unittest
from unittest.mock import patch, MagicMock
from src.rag_model.advanced_rag_evaluation import AdvancedRAGEvaluator, TEST_QUERIES_WITH_GROUND_TRUTH

class TestAdvancedRAGEvaluator(unittest.TestCase):

    def setUp(self):
        self.evaluator = AdvancedRAGEvaluator(output_dir="test_results")

    @patch('src.rag_model.advanced_rag_evaluation.ChatOpenAI')
    @patch('src.rag_model.advanced_rag_evaluation.OpenAIEmbeddings')
    def test_initialize_evaluator(self, MockEmbeddings, MockChatOpenAI):
        self.assertEqual(self.evaluator.output_dir, "test_results")
        self.assertEqual(self.evaluator.test_queries, TEST_QUERIES_WITH_GROUND_TRUTH)
        self.assertIsNotNone(self.evaluator.evaluation_llm)
        self.assertIsNotNone(self.evaluator.embeddings)

    def test_evaluate_faithfulness(self):
        query = "How much protein should I consume daily for muscle growth?"
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        contexts = ["For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."]
        score = self.evaluator.evaluate_faithfulness(query, response, contexts)
        self.assertIsInstance(score, float)

    def test_evaluate_answer_relevancy(self):
        query = "How much protein should I consume daily for muscle growth?"
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_answer_relevancy(query, response)
        self.assertIsInstance(score, float)

    def test_evaluate_context_relevancy(self):
        query = "How much protein should I consume daily for muscle growth?"
        contexts = ["For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."]
        score = self.evaluator.evaluate_context_relevancy(query, contexts)
        self.assertIsInstance(score, float)

    def test_evaluate_context_precision(self):
        query = "How much protein should I consume daily for muscle growth?"
        contexts = ["For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."]
        score = self.evaluator.evaluate_context_precision(query, contexts)
        self.assertIsInstance(score, float)

    def test_evaluate_fitness_domain_accuracy(self):
        query = "How much protein should I consume daily for muscle growth?"
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_fitness_domain_accuracy(query, response)
        self.assertIsInstance(score, float)

    def test_evaluate_scientific_correctness(self):
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_scientific_correctness(response)
        self.assertIsInstance(score, float)

    def test_evaluate_practical_applicability(self):
        query = "How much protein should I consume daily for muscle growth?"
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_practical_applicability(query, response)
        self.assertIsInstance(score, float)

    def test_evaluate_safety_consideration(self):
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_safety_consideration(response)
        self.assertIsInstance(score, float)

    def test_evaluate_retrieval_precision(self):
        query = "How much protein should I consume daily for muscle growth?"
        contexts = ["For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."]
        score = self.evaluator.evaluate_retrieval_precision(query, contexts)
        self.assertIsInstance(score, float)

    def test_evaluate_retrieval_recall(self):
        query = "How much protein should I consume daily for muscle growth?"
        contexts = ["For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."]
        ground_truth = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_retrieval_recall(query, contexts, ground_truth)
        self.assertIsInstance(score, float)

    def test_evaluate_answer_completeness(self):
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        ground_truth = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_answer_completeness(response, ground_truth)
        self.assertIsInstance(score, float)

    def test_evaluate_answer_conciseness(self):
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_answer_conciseness(response)
        self.assertIsInstance(score, float)

    def test_evaluate_answer_helpfulness(self):
        query = "How much protein should I consume daily for muscle growth?"
        response = "For muscle growth, consume 1.6-2.2g of protein per kg of bodyweight daily."
        score = self.evaluator.evaluate_answer_helpfulness(query, response)
        self.assertIsInstance(score, float)

if __name__ == '__main__':
    unittest.main()
