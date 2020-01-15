This version was designed to run on Python 3.7.4.


Data structure :

    Rendu
    |
    |- /Public (contains data)
    |
    |- /weights (CNN weights)
    |
    |- /Old_Techniques
    |
    |- /Small_Cnn
    |
"rest of the files for predictions"


    /Public          contains the data, it is the output of the unziped file obtained on the challenge page. It also contains the calatog with the labels for the lenses.
    /weights         contains the output of the CNNs. we will try to include pre-trained weights if the size allows it. (else it will be on our github)
    /Old_techniques  contains the try to make kPCA and SVM work. Very messy, was abandonned almost immediatly.
    /Small_Cnn       contains the small CNN we learned.


    "rest of the files for predicitons" is composed of :

        Efficient_net.py,       the main python code used to train the huge network. We cannot provided a true code to reproduce exactly the ouptuts, it was trained on multiple computers, multiple times.
                                As a simple test, you could set the crit pixel to a large value, set rate_expansion to 0 and learn for 30 epochs. This will only detect large lenses, with about 92% accuracy.
                                For a more detailed version, we trained first the whole code with decresing lens sizes. Then, the classifier was trained, and then, the lowere layers of the CNN step by step.

        Eff_net_predictions.py  The pre-trained weights applied to the Effiecient Net CNN. This will output a prediction, and print the labels in the terminal with the number of false positive / negative.
                                Due to the fact that the final challenge was not out up to the deadline of the report, we did not make any prediction saved as a .csv file. It will be done for the final challenge.
                                This should work fine, once the data is imported in Public (and the librairies installed).

        CNN.py                  Contains all the different keras CNN models tested during the period of the project. They can be imported directly into the two .py above, as model = CNN_name().

        herlpers_bis_effnet.py  Contains most of the little functions that were not mandatory in the main executable. These include some of the logarithmic strech algorithms, methods to load the data and the labels.
                                There are also function to plot the images, to generate more background, etc ...



Librairies :

    IMPORTANT : Due to the large amount of data to process, there is a lot of different libraries needed to make the code work.
    You can directly check the librairies needed in the headers of the 3 python files. Here is a list of everything in use, and a quick explanation.

    numpy           for all the data transfer between arrays, processing, leraning.
    tensorflow      for the machine learning (obviously). TF (version > 2.0) contains keras here, so we don't need to import it.
    Pil, c2v        for some image manipulation.
    Matplotlib      for plots
    sklearn         to plot the ROC AUC curve
    Astropy         a python tool to load .fits files, strech them, and some preprocessing.
    efficientnet    the keras application that implements Efficient_net as a keras model. The only "external library used" : can be obtained at https://github.com/Tony607/efficientnet_keras_transfer_learning

    If the base packages are loaded, the only things that have to be added is the astropy librairy and the keras implementation of Efficient Net.


Raw Data :

    The data needs to be Downloaded from the website : http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html.
    When unzipped, it will be presented as a folder Public, it has to be remplace by the placeholder in our "Rendu". Simply drag and drop and then overwrite Public.
    Then, the catalog has to be Downloaded, and displaced into Public as well.
    Once the librairies are set up, the code should be running.
    We will try to include a small sample of Images, to prevent the long download times. If the time allows we will put a sample of images on our github (~1000) for easier download.
    Normally there is 69 images, the first of the set, to check that everything is working fine. If you have time you can add more with the procedure above.


Parameters (in Efficient_net.py) :

    There are a couple parameters in the main code :

    Checkpoint_network       do you want to restore the network from existing weights ? (weights are always saved after a run).
    Net_file_path            location of the weights.
    Test_set_fresh           At the end of the simulation, test on a new data set.
    Image_load_length        Length of the images you want to feed the NN.
    start_train              The location to start loading the image, practical to make a sweep on multiple files batches.
    Path                     Path to the Images / Labels.
    label_name               Name of the catalog containing the labels.
    train_ratio              Ration to split train and test set.
    EPOCHS                   Number of epochs done by the training set.
    Images_load_length_test  Lenght of the images to load for the fresh test set
    start_test               start of the load of train images it is better to have it pretty far (at 50000 for example) to avoir overlap with the training.
    Crit_Pixels              Parameter to choose if lenses are flagged as 1 or 0 as function of the number of pixel lensed.
    ext_rate                 When training with low crit pixels, the data set starts to have a lot of ones. The extension rate will load more zeros to take the effect into account.


    Parameters in Eff_net_predictions.py are the same, but not all the parameters are present
