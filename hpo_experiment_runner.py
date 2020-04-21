import os
import ray
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import threading

ray.init()


def run(configured_hpo_instance, save_file_path=None):
    best_result, results = configured_hpo_instance.execute(result_added_hook=add_result_to_graph)

    if save_file_path is not None:
        results.save(os.path.join(os.getcwd(), save_file_path))

    return best_result, results


fig, a = plt.subplots(2, 2)
raw_scores_axis = a[0][0]
average_score_axis= a[0][1]
prev_accuracy_over_epochs = a[1][0]
prev_loss_over_epochs = a[1][1]

raw_scores = list()
average_scores = list()
total = 0
count = 0
prev_iteration_history = None


def add_result_to_graph(result):
    global prev_iteration_history
    global total
    global count

    if result.score() is None:
        return

    raw_scores.append(result.score())
    total += result.score()
    count += 1
    average_scores.append(total / count)
    prev_iteration_history = result.training_history()


def update_graph(i):
    raw_scores_axis.clear()
    raw_scores_axis.set_title("Score for each result.")
    raw_scores_axis.plot(raw_scores)
    average_score_axis.clear()
    average_score_axis.set_title("Average score over optimisation period.")
    average_score_axis.plot(average_scores)
    prev_accuracy_over_epochs.clear()
    prev_accuracy_over_epochs.set_title("Last iteration - Accuracy over epochs.")
    prev_loss_over_epochs.clear()
    prev_loss_over_epochs.set_title("Last iteration - Loss over epochs.")
    if prev_iteration_history is not None:
        prev_accuracy_over_epochs.plot(prev_iteration_history["accuracy"])
        prev_accuracy_over_epochs.plot(prev_iteration_history["val_accuracy"])
        prev_accuracy_over_epochs.legend(["Training Accuracy", "Validation Accuracy"])
        prev_loss_over_epochs.plot(prev_iteration_history["loss"])
        prev_loss_over_epochs.plot(prev_iteration_history["val_loss"])
        prev_loss_over_epochs.legend(["Training Loss", "Validation Loss"])


def run_with_live_graphs(configured_hpo_instance, save_file_path=None):
    def run_hpo():
        global best_result
        global results

        best_result, results = configured_hpo_instance.execute(result_added_hook=add_result_to_graph)

    hpo_thread = threading.Thread(target=run_hpo)
    hpo_thread.start()

    ani = animation.FuncAnimation(fig, update_graph, interval=1000)
    plt.show()

    hpo_thread.join()

    if save_file_path is not None:
        results.save(os.path.join(os.getcwd(), "results.res"))

    return best_result, results
