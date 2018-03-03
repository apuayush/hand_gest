from modules import *

def load_test_image(path='/home/apurvnit/datasets/gestures_test/Marcel-Test'):
    c=1
    all_data = os.listdir(path)
    try:
        all_data.remove('MiniTrieschGallery')
    except: pass
    nb_classes = len(all_data)
    classes = {
            'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4,
            'F':5
            }

    X = []
    Y = []
    for gesture in all_data:
        sub_path = os.path.join(path,gesture)
        for subfolder in os.listdir(sub_path):
            g_path = os.path.join(sub_path, subfolder)
            for image in os.listdir(g_path):
                img_path = os.path.join(g_path, image)
                img = image_processor(img_path)
                X.append(img)
                test_y = [0.] * nb_classes
                test_y[classes[gesture]] = 1.0
                Y.append(np.array(test_y))
    X = np.array(X)/255
    Y = np.array(Y)
    return X, Y


def load_train_image(path='/home/apurvnit/datasets/gestures_test/Marcel-Train'):
    c=1
    all_data = os.listdir(path)
    try:
        all_data.remove('MiniTrieschGallery')
    except: pass
    nb_classes = len(all_data)
    classes = {
            'A':0,
            'B':1,
            'C':2,
            'D':3,
            'E':4,
            'F':5
            }

    X = []
    Y = []
    for gesture in all_data:
        g_path = os.path.join(path, gesture)
        for image in os.listdir(g_path):
            img_path = os.path.join(g_path, image)
            img = image_processor(img_path)
            X.append(np.array(img))
            test_y = [0.] * nb_classes
            test_y[classes[gesture]] = 1.0
            Y.append(np.array(test_y))
    X = np.array(X)/255
    Y = np.array(Y)
    return X, Y


def image_processor(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200), interpolation = cv2.INTER_NEAREST).astype(np.float32)
    print(img.shape)
    img = img.flatten().reshape(200,200,1)

    return img


    