import os

def get_class_counts(base_dir):
    class_counts = {}
    for lang in os.listdir(base_dir):
        lang_path = os.path.join(base_dir, lang)
        if os.path.isdir(lang_path):
            count = len(os.listdir(lang_path))
            class_counts[lang] = count
    return class_counts

base_dir = "C:/Users/Admin/Desktop/internship_projects/language_predictor/clened_dataset"
counts = get_class_counts(base_dir)
print("Class counts:", counts)
print("Target size:", max(counts.values()))
