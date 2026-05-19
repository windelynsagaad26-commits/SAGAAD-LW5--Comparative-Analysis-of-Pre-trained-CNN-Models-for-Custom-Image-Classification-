# SAGAAD-LW5--Comparative-Analysis-of-Pre-trained-CNN-Models-for-Custom-Image-Classification-

# Google Colab Link: https://colab.research.google.com/drive/18eLo3RIXTf9NeY3bVeBSbdCvxBw4AHHT?usp=sharing


### 📝 FINAL REFLECTION: MODEL EVALUATION & ANALYSIS
### A. Model Performance
Highest Accuracy: Ang Custom MobileNetV2 ang nag-una sa tanan base sa comparison_df (Test Acc: 0.0615). Nahimo kini tungod sa iyang arkitektura nga naggamit og inverted residual blocks ug depthwise separable convolutions, nga haom kaayo sa pag-extract og importanteng features bisan pa man og limitado ang gidak-on sa atong custom dataset kon itandi sa mas bug-at nga VGG16.

#### Lowest Performance: Ang Custom VGG16 ang nagpakita og pinakaubos nga abilidad sa pagsulbad sa maong buluhaton. Tungod kay dako ug parameter-heavy kini nga modelo, dali kini maapektuhan sa vanishing gradients ug nanginahanglan og mas daghan pa nga epochs aron mag-converge, ilabi na nga gamay ra atong gigamit nga learning rate (0.0001) para sa transfer learning.

#### Loss Comparison: Namatikdan nga mas taas ang test loss sa mga custom models kon itandi sa ilang train loss. Kini nagpakita og dakong generalization gap—pasabot nga maayo ra sila sa nailhan na nga data apan naglisod sa bag-ong mga imahe. Masulbad kini pinaagi sa pagpataas sa epochs o pag-apply og data augmentation.

### B. Evaluation Metrics
Accuracy vs. Others: Dili paigo ang Accuracy kay gilangkuban man og 20 ka nagkalain-laing klase sa tanom ang atong dataset. Kung naay usa ka klase nga mas daghan og sulod, pwede mopataas ang accuracy pinaagi lang sa pagtagna sa maong klase. Ang Precision ug Recall maoy naghatag og kasigurohan nga matag-usa gyud ka matang sa kahoy na-evaluate og sakto.

Best F1-score: Ang Custom MobileNetV2 gihapon ang nag-una sa F1-score (0.0244). Nagpasabot kini nga mas balanse ug lig-on ang relasyon sa iyang Precision ug Recall kon itandi sa ubang mga modelo nga gi-test.

Precision/Recall Differences: Adunay mga modelo nga taas og Recall sa partikular nga mga klase sama sa 'Willow Tree' ug 'Pear Tree' apan mubo ang Precision. Pasabot niini, sige og "pataka" og tagna ang modelo sa maong mga klase (taas ang False Positives) bisan og lahi nga kahoy ang naa sa imahe.

###  C. Confusion Matrix Analysis
Misclassifications: Base sa confusion matrices, klaro nga naglibog ang mga modelo sa pag-ila sa mga kahoy nga adunay halos magkaparehas nga porma sa dahon, estruktura, o panit sa punoan.

Patterns: Naay namatikdan nga "vertical banding" o patindog nga linya sa matrix. Timaan kini nga ang usa ka modelo nagsige na lang og pusta o prediksyon sa usa ka 'majority class' matag higayon nga magduhaduha kini sa iyang tubag.

### D. ROC and AUC
Highest AUC: Ang Custom MobileNetV2 gihapon ang nagkupot sa pinakataas nga score (AUC: 0.5489).

AUC Significance: Ang bili sa AUC mao ang pagpakita sa katakos sa modelo nga mas mopili o mo-ranggo sa saktong klase (positive example) kaysa sa sayop (negative example). Tungod kay lapas man kini sa 0.5, nagpasabot nga mas maayo pa gihapon ang iyang performance kaysa sa sulog-sulog o sulong tagna (random guessing).

### E. Explainability (Grad-CAM)
Revelations: Pinaagi sa Grad-CAM, nakita nato nga ang MobileNetV2 mas nag-focus sa detalye ug texture sa mga dahon, samtang ang EfficientNetB0 usahay mabalhin ang atensyon ngadto sa background o sa kilid-kilid sa imahe.

Focus: Ang mga modelo nga mas taas og accuracy nagpakita og mga heatmaps nga saktong nakatutok o nakasentro sa mismong lawas sa kahoy o sa iyang mga dahon.

Meaningful Heatmaps: Bisan pa og limpyo ug klaro ang heatmaps sa VGG16 tungod sa iyang diretso nga arkitektura, ang MobileNetV2 gihapon ang nagpakita og mga activations nga mas naay lohikal nga koneksyon sa saktong klasipikasyon.

### F. Model Comparison & Improvement
Recommendation: Mapili nako ang MobileNetV2. Gawas nga maayo kini og accuracy sa atong testing, episyente ug gaan kaayo ang gidak-on sa iyang file, nga angayan kaayo para i-deploy sa mga mobile devices.

Improvements: Aron mapalambo pa kini, mahimong mogamit og Data Augmentation (sama sa pag-rotate ug pag-zoom sa mga hulagway), pagpataas sa pag-train ngadto sa 30-50 Epochs, ug paghimo og Fine-tuning pinaagi sa pag-unfreeze sa kataposang mga convolutional layers sa base model aron mas mo-adapt sa atong dataset.

### G. Real-World Application
Application: Magamit kini isip kasingkasing sa usa ka mobile application para sa mga botanist, foresters, o mga nature lovers aron dali ra maka-identify og mga espisye sa kahoy samtang naa sa lasang.

Risks: Kung dili sakto ang modelo, risgo kini kay basin masayop og ilha ang mga invasive nga tanom nga makadaot sa kinaiyahan, o makahatag og sayop nga impormasyon sa mga ecological surveys.

Integration: Ang atong na-save nga .keras models mahimo natong i-convert ngadto sa TensorFlow Lite (.tflite) nga format. Human niini, pwede na kini isulod ug daganon sa usa ka mobile app gamit ang mga frameworks sama sa Flutter o React Native.

