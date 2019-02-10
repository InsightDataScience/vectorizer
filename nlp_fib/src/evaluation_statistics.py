from sklearn.metrics import accuracy_score

__author__ = "Pujaa Rajan"
__email__ = "pujaa.rajan@gmail.com"

class Evaluation:
    """ This class takes in the probability files
        """

    def __init__(self, ngram_test, output_file_path):
        self.correct_answer = ngram_test.test_fill_in_the_blank['answer']
        self.forward_ngram_model_answer = ngram_test.test_fill_in_the_blank['Forward Answers']
        self.backward_ngram_model_answer = ngram_test.test_fill_in_the_blank['Backward Answers']

        self.forward_accuracy = self.calculate_accuracy(self.correct_answer, self.forward_ngram_model_answer)
        self.backward_accuracy = self.calculate_accuracy(self.correct_answer, self.backward_ngram_model_answer)
        self.merged_accuracy = None # TODO
        self.create_summary_statistics(self.forward_accuracy, self.backward_accuracy, output_file_path)

    def calculate_accuracy(self, correct_answer, model_answer):
        correct_count = 0
        correct_answer_index = 0
        correct_answer_list = correct_answer.tolist()
        for answer_list in model_answer.tolist():
            if correct_answer_list[correct_answer_index] in answer_list:
                correct_count = correct_count + 1
            correct_answer_index = correct_answer_index+1
        return correct_count/len(correct_answer)


    def create_summary_statistics(self, forward_accuracy, backward_accuracy, output_file_path):

        summary_statistics = open(f'summary_statistics.txt', 'a')
        summary_statistics.write(f'The forward model has an accuracy of: {forward_accuracy}\n')
        summary_statistics.write(f'The backward model has an accuracy of: {backward_accuracy}\n')
        summary_statistics.close()
