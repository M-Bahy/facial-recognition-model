from deepface import DeepFace
import matplotlib.pyplot as plt
import threading
import dotenv
import os


dotenv.load_dotenv()


def display_images(image_1, image_2, image_3):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image_1)
    axs[0].set_title("Image 1")
    axs[1].imshow(image_2)
    axs[1].set_title("Image 2")
    axs[2].imshow(image_3)
    axs[2].set_title("Image 3")
    plt.show()


def print_results(image_1_path, image_2_path, image_3_path):
    print("Is 1 similar to 2 ?")
    print(DeepFace.verify(image_1_path, image_2_path)["verified"])
    print("Is 1 similar to 3 ?")
    print(DeepFace.verify(image_1_path, image_3_path)["verified"])
    print("Is 2 similar to 3 ?")
    print(DeepFace.verify(image_2_path, image_3_path)["verified"])


def search(image_path):
    df = DeepFace.find(img_path=image_path, db_path="dataset/")
    print(df)


def main(display_images, print_results):
    image_1_path = os.getenv("image_1")
    image_2_path = os.getenv("image_2")
    image_3_path = os.getenv("image_3")

    image_1 = DeepFace.detectFace(image_1_path)
    image_2 = DeepFace.detectFace(image_2_path)
    image_3 = DeepFace.detectFace(image_3_path)

    thread_1 = threading.Thread(
        target=display_images, args=(image_1, image_2, image_3)
    )
    thread_2 = threading.Thread(
        target=print_results, args=(image_1_path, image_2_path, image_3_path)
    )

    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    search(image_1_path)


if __name__ == "__main__":
    main(display_images, print_results)


"""
Model	        LFW Score	YTF Score

Facenet512	    99.65%	    -
SFace	        99.60%	    -
ArcFace 	    99.41%	    -
Dlib	        99.38 %	    -
Facenet	        99.20%	    -
VGG-Face	    98.78%	    97.40%      <--- This is the model we are using
Human-beings	97.53%	    -
OpenFace	    93.80%	    -
DeepID	-	    97.05%

"""
