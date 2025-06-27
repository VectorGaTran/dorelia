from fastai.vision.all import *
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pathlib
import platform
import multiprocessing # Importă multiprocessing

# --- Configurație ---
MODEL_EXPORT_PATH = Path("models/architectural_style_model_fastai_resnet50_1.pkl")
TEST_IMAGE_FOLDER = Path("personal test") 
def main(): # Întregul tău cod va merge în această funcție
    print(f"Se încarcă modelul de la: {MODEL_EXPORT_PATH}")

    if platform.system() == "Windows":
        original_posix_path = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            print("Patch PosixPath aplicat pentru Windows.")
            learn = load_learner(MODEL_EXPORT_PATH)
        finally:
            pathlib.PosixPath = original_posix_path
            print("Patch PosixPath restabilit.")
    else:
        learn = load_learner(MODEL_EXPORT_PATH)

    try:
        if 'learn' not in locals():
            learn = load_learner(MODEL_EXPORT_PATH)

        print("Modelul a fost încărcat cu succes.")
        print("Imaginile vor fi redimensionate și normalizate automat după configurația modelului.")

    except FileNotFoundError:
        print(f"Eroare: Nu a fost găsit modelul la {MODEL_EXPORT_PATH}.")
        exit()
    except Exception as e:
        print(f"Eroare la încărcarea modelului: {e}")
        import traceback
        traceback.print_exc()
        exit()

    if not TEST_IMAGE_FOLDER.is_dir(): 
        print(f"Eroare: directorul cu imagini pentru testare nu a fost găsit sau nu este un director la '{TEST_IMAGE_FOLDER}'.")
        print(f"Calea absolută verificată: {TEST_IMAGE_FOLDER.resolve()}")
        print("Verifică dacă ai creat acest folder și ai specificat calea corectă.")
        test_image_files = []
    else:
        test_image_files = get_image_files(TEST_IMAGE_FOLDER)

    # if not test_image_files:
    #     print(f"Nu au fost găsite imagini în {TEST_IMAGE_FOLDER}.")
    # else:
    #     print(f"Au fost găsite {len(test_image_files)} imagini în {TEST_IMAGE_FOLDER}.")
    #     print("Se creează DataLoader pentru testare...")
    #     try:
    #         # test_dl = learn.dls.test_dl(test_image_files, with_labels=True, bs=learn.dls.bs, num_workers=0)
    #         test_dl = learn.dls.test_dl(test_image_files, with_labels=True, bs=learn.dls.bs)
    #         print(f"Dataloader pentru testare creat pentru {len(test_dl.dataset)} imagini și dimensiunea lotului {test_dl.bs}.")

    #         print("\n--- Evaluarea pe setul de testare ---")
    #       # preds: predicțiile modelului (probabilități)
    #       # targs: indicele stilurilor reale
    #       # decoded_preds: stilul prezis cu cea mai mare probabilitate
    #         preds, targs, decoded_preds = learn.get_preds(dl=test_dl, with_decoded=True)

    #         test_accuracy = accuracy(preds, targs)
    #         print(f"Acuratețea setului de testare: {test_accuracy.item():.4f}")

    #         k_val = 3
    #         if preds.shape[1] >= k_val:
    #             test_topk_accuracy = top_k_accuracy(preds, targs, k=k_val)
    #             print(f"Acuratețea setului de testare Top-{k_val}: {test_topk_accuracy.item():.4f}")
    #         else:
    #             print(f"Nu se poate calcula Top-{k_val} acuratețea deoarece sunt mai puține de {k_val} clase în predicții ({preds.shape[1]}).")

    #         interp_test = ClassificationInterpretation.from_learner(learn, dl=test_dl)

    #         print("\nSe generează matricea de confuzie...")
    #         try:
    #             interp_test.plot_confusion_matrix(figsize=(12,12), dpi=60)
    #             plt.title("Matricea de confuzie (auto-generată)", fontsize=16)
    #             plt.tight_layout() 
    #             plt.show()
    #         except Exception as e_conf_matrix:
    #             print(f"Eroare la plotarea matricei de confuzie: {e_conf_matrix}")
    #             import traceback
    #             traceback.print_exc()

    #         if hasattr(learn, 'dls') and learn.dls and hasattr(learn.dls, 'vocab'):
    #             target_names = learn.dls.vocab
    #             print("\Raport legat de clasificarea stului de testare:")
    #             try:
    #                 labels_for_report = range(len(target_names))
    #                 print(classification_report(
    #                     targs.numpy(),
    #                     decoded_preds.numpy(),
    #                     labels=labels_for_report, 
    #                     target_names=target_names,
    #                     zero_division=0
    #                 ))
    #             except Exception as e_report:
    #                 print(f"Eroare la generarea raportului: {e_report}")
    #                 import traceback
    #                 traceback.print_exc()

    #     except RuntimeError as e_rt:
    #         if "An attempt has been made to start a new process" in str(e_rt):
    #             print(f"\n A apărut eroarea de multiprocessing: {e_rt}")
    #         else:
    #             print(f"\n A apărut o eroare RuntimeError în timpul evaluării: {e_rt}")
    #             import traceback
    #             traceback.print_exc()
    #     except Exception as e_eval:
    #         print(f"\n A apărut o eroare generală în timpul evaluării: {e_eval}")
    #         import traceback
    #         traceback.print_exc()
    #         print(f"Asigură-te că '{TEST_IMAGE_FOLDER}' are subdirectoare numite după clasele tale (de ex., {TEST_IMAGE_FOLDER}/ArtDeco).")

###################################################################################################################################
    print("\n--- Exemplu: Predicții pe imagini specifice (Top K stiluri) ---")
    individual_test_image_paths = []
    K = 1 # Numărul de stiluri de top pe care dorești să le afișezi
    print(learn.model)
    if test_image_files and len(test_image_files) > 0:
        num_example_images = min(15, len(test_image_files)) # Limitează numărul de imagini pentru exemplu
        individual_test_image_paths = test_image_files[:num_example_images]
        print(f" Au fost selectate primele {len(individual_test_image_paths)} imagini din '{TEST_IMAGE_FOLDER}' pentru exemple de predicții individuale:")
    else:
        print(f" Nu au fost găsite imagini în {TEST_IMAGE_FOLDER}.")

    if individual_test_image_paths:
        print(f"\nPredicții pe imagini individuale (Top {K} stiluri):")
        for img_path in individual_test_image_paths:
            if not img_path.is_file():
                print(f" Avertisment: Fișierul imagine nu a fost găsit la {img_path}, se omite.")
                continue
            try:
                # learn.predict() returnează: clasa prezisă, indexul clasei prezise, și output-urile (probabilitățile)
                pred_class, pred_idx, outputs = learn.predict(img_path)

                print(f"Imagine: {img_path.name}")

                # Obține cele mai mari K probabilități și indicii lor
                top_k_probs, top_k_indices = torch.topk(outputs, k=K)

                for i in range(K):
                    idx = top_k_indices[i].item() 
                    prob = top_k_probs[i].item()  
                    style_name = learn.dls.vocab[idx] 
                    print(f"   {i+1}. Stil: {style_name} (Încredere: {prob:.4f})")
                print("-" * 30) 

            except Exception as e:
                print(f"Nu s-a putut prezice pentru {img_path}: {e}")
    else:
        print("\nNu au fost selectate sau furnizate imagini specifice pentru exemplele de predicții individuale.")

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main() # Apelează funcția principală