# T013-I complete linear capacity tables

Fixed source arrays, float64 algebra, no model execution or optimizer. These are gradient predictions, not AP or finite-step performance.

## Fold fit numerical information

| Fold | Rank | Tolerance | B Frobenius norm | All singular condition | Retained condition | Training SSE |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 16 | 7.22336978527e-15 | 2679.07220222 | 4128.54626728 | 4128.54626728 | 7.81396522422 |
| 1 | 16 | 6.67510676886e-15 | 1722.68011902 | 3548.24318109 | 3548.24318109 | 5.77459654755 |
| 2 | 16 | 6.72271038929e-15 | 3140.09432065 | 2907.71375202 | 2907.71375202 | 6.01806438713 |
| 3 | 16 | 6.78953025659e-15 | 3327.31975822 | 2936.07342221 | 2936.07342221 | 5.73109272502 |

## Fold 0 singular values

| Index | Value |
| --- | --- |
| 0 | 0.677732614027 |
| 1 | 0.437718011003 |
| 2 | 0.137723948434 |
| 3 | 0.0557179968697 |
| 4 | 0.0126693844674 |
| 5 | 0.00845929622173 |
| 6 | 0.00517663666002 |
| 7 | 0.00388737548607 |
| 8 | 0.00343062948538 |
| 9 | 0.00165704830745 |
| 10 | 0.00121957368327 |
| 11 | 0.000593834864025 |
| 12 | 0.000543432031323 |
| 13 | 0.0003032393556 |
| 14 | 0.000252756673198 |
| 15 | 0.000164157688966 |

## Fold 1 singular values

| Index | Value |
| --- | --- |
| 0 | 0.626291840768 |
| 1 | 0.495131582729 |
| 2 | 0.131057704451 |
| 3 | 0.0577560605708 |
| 4 | 0.0209607560315 |
| 5 | 0.0104260715332 |
| 6 | 0.00603214677579 |
| 7 | 0.00379327705088 |
| 8 | 0.00293116964726 |
| 9 | 0.00189215882043 |
| 10 | 0.00130800361592 |
| 11 | 0.00067607811662 |
| 12 | 0.000585578178264 |
| 13 | 0.000361905360415 |
| 14 | 0.000213815044585 |
| 15 | 0.000176507586657 |

## Fold 2 singular values

| Index | Value |
| --- | --- |
| 0 | 0.630758250086 |
| 1 | 0.450109416457 |
| 2 | 0.170072284618 |
| 3 | 0.054149749141 |
| 4 | 0.0174317309196 |
| 5 | 0.0085258551877 |
| 6 | 0.00759127070422 |
| 7 | 0.00376330358019 |
| 8 | 0.00280771382257 |
| 9 | 0.00206487634157 |
| 10 | 0.00129663095863 |
| 11 | 0.000577643417848 |
| 12 | 0.000504562904291 |
| 13 | 0.000401637638535 |
| 14 | 0.000258941066136 |
| 15 | 0.00021692584067 |

## Fold 3 singular values

| Index | Value |
| --- | --- |
| 0 | 0.637027623617 |
| 1 | 0.493610338779 |
| 2 | 0.170437073466 |
| 3 | 0.0554739236848 |
| 4 | 0.0201423143512 |
| 5 | 0.0095760972541 |
| 6 | 0.00765896613283 |
| 7 | 0.00346946094439 |
| 8 | 0.00331752118017 |
| 9 | 0.00175878567446 |
| 10 | 0.00135645847974 |
| 11 | 0.000708211415196 |
| 12 | 0.00065221806029 |
| 13 | 0.000405555519225 |
| 14 | 0.000222400240985 |
| 15 | 0.000216965835663 |

## Residual statistics by scope and reference

| Scope | Method | N | SSE | Residual energy | R2 | Median residual cosine | Positive residual dot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | linear | 64 | 19.3573396009 | 14.0206479088 | -0.380630890016 | 0.0683647559385 | 33 |
| overall | common | 64 | 14.0206479088 | 14.0206479088 | 0 | 0 | 0 |
| overall | covariance | 64 | 14.009453319 | 14.0206479088 | 0.000798435981327 | 0.224351387029 | 40 |
| clean | linear | 32 | 7.54503118743 | 2.09734316375 | -2.59742331052 | -0.0225429079024 | 15 |
| clean | common | 32 | 2.09734316375 | 2.09734316375 | 0 | 0 | 0 |
| clean | covariance | 32 | 2.09673916223 | 2.09734316375 | 0.000287984117384 | 0.351928061348 | 22 |
| corrupted | linear | 32 | 11.8123084135 | 11.923304745 | 0.00930919186707 | 0.184326817812 | 18 |
| corrupted | common | 32 | 11.923304745 | 11.923304745 | 0 | 0 | 0 |
| corrupted | covariance | 32 | 11.9127141568 | 11.923304745 | 0.000888225913746 | 0.0421193651033 | 18 |
| fold0 | linear | 16 | 4.11774053898 | 0.569842349942 | -6.22610479793 | 0.257425658925 | 9 |
| fold0 | common | 16 | 0.569842349942 | 0.569842349942 | 0 | 0 | 0 |
| fold0 | covariance | 16 | 0.569868489999 | 0.569842349942 | -4.58724357819e-05 | 0.336732312809 | 10 |
| fold0/clean | linear | 8 | 2.20895454609 | 0.138673777561 | -14.9291438147 | 0.257425658925 | 5 |
| fold0/clean | common | 8 | 0.138673777561 | 0.138673777561 | 0 | 0 | 0 |
| fold0/clean | covariance | 8 | 0.138361707986 | 0.138673777561 | 0.00225038633809 | 0.467967578517 | 6 |
| fold0/corrupted | linear | 8 | 1.90878599289 | 0.431168572382 | -3.42700631529 | -0.015172547795 | 4 |
| fold0/corrupted | common | 8 | 0.431168572382 | 0.431168572382 | 0 | 0 | 0 |
| fold0/corrupted | covariance | 8 | 0.431506782013 | 0.431168572382 | -0.000784402326009 | -0.16561627713 | 4 |
| fold1 | linear | 16 | 5.74325407315 | 5.04487813571 | -0.13843266748 | -0.386635180116 | 6 |
| fold1 | common | 16 | 5.04487813571 | 5.04487813571 | 0 | 0 | 0 |
| fold1 | covariance | 16 | 5.04066067955 | 5.04487813571 | 0.000835987718903 | -0.36881014915 | 5 |
| fold1/clean | linear | 8 | 1.51375212412 | 0.57295983945 | -1.64198643586 | -0.29560145445 | 3 |
| fold1/clean | common | 8 | 0.57295983945 | 0.57295983945 | 0 | 0 | 0 |
| fold1/clean | covariance | 8 | 0.573859694912 | 0.57295983945 | -0.00157053845573 | -0.817445777158 | 2 |
| fold1/corrupted | linear | 8 | 4.22950194903 | 4.47191829626 | 0.0542085814568 | -0.398841334097 | 3 |
| fold1/corrupted | common | 8 | 4.47191829626 | 4.47191829626 | 0 | 0 | 0 |
| fold1/corrupted | covariance | 8 | 4.46680098463 | 4.47191829626 | 0.00114432135993 | -0.229736261725 | 3 |
| fold2 | linear | 16 | 4.88041931739 | 4.22724278649 | -0.154515972677 | 0.20016464598 | 10 |
| fold2 | common | 16 | 4.22724278649 | 4.22724278649 | 0 | 0 | 0 |
| fold2 | covariance | 16 | 4.22412969483 | 4.22724278649 | 0.00073643550056 | 0.394705298598 | 12 |
| fold2/clean | linear | 8 | 3.06201793923 | 0.851868718034 | -2.59447162973 | -0.0406465487855 | 4 |
| fold2/clean | common | 8 | 0.851868718034 | 0.851868718034 | 0 | 0 | 0 |
| fold2/clean | covariance | 8 | 0.85095007554 | 0.851868718034 | 0.00107838505449 | 0.547717162834 | 7 |
| fold2/corrupted | linear | 8 | 1.81840137816 | 3.37537406846 | 0.461274116208 | 0.503524276631 | 6 |
| fold2/corrupted | common | 8 | 3.37537406846 | 3.37537406846 | 0 | 0 | 0 |
| fold2/corrupted | covariance | 8 | 3.37317961929 | 3.37537406846 | 0.000650135101779 | 0.166305271549 | 5 |
| fold3 | linear | 16 | 4.61592567136 | 4.17868463662 | -0.104636045255 | 0.00771444991691 | 8 |
| fold3 | common | 16 | 4.17868463662 | 4.17868463662 | 0 | 0 | 0 |
| fold3 | covariance | 16 | 4.17479445461 | 4.17868463662 | 0.000930958505966 | 0.288460168343 | 13 |
| fold3/clean | linear | 8 | 0.760306577983 | 0.533840828702 | -0.424219612111 | -0.184592663786 | 3 |
| fold3/clean | common | 8 | 0.533840828702 | 0.533840828702 | 0 | 0 | 0 |
| fold3/clean | covariance | 8 | 0.533567683789 | 0.533840828702 | 0.000511659839836 | 0.351928061348 | 7 |
| fold3/corrupted | linear | 8 | 3.85561909338 | 3.64484380791 | -0.0578283450732 | 0.192109459654 | 5 |
| fold3/corrupted | common | 8 | 3.64484380791 | 3.64484380791 | 0 | 0 | 0 |
| fold3/corrupted | covariance | 8 | 3.64122677082 | 3.64484380791 | 0.000992370944797 | 0.224351387029 | 6 |
| case/clean_s0 | linear | 32 | 7.54503118743 | 2.09734316375 | -2.59742331052 | -0.0225429079024 | 15 |
| case/clean_s0 | common | 32 | 2.09734316375 | 2.09734316375 | 0 | 0 | 0 |
| case/clean_s0 | covariance | 32 | 2.09673916223 | 2.09734316375 | 0.000287984117384 | 0.351928061348 | 22 |
| case/color_cast_s2 | linear | 8 | 1.58072927522 | 0.124965682548 | -11.6493069376 | -0.127328837355 | 4 |
| case/color_cast_s2 | common | 8 | 0.124965682548 | 0.124965682548 | 0 | 0 | 0 |
| case/color_cast_s2 | covariance | 8 | 0.123957374357 | 0.124965682548 | 0.00806868069684 | 0.282724513085 | 4 |
| case/contrast_s2 | linear | 8 | 2.80194576629 | 2.80065865176 | -0.000459575655253 | 0.380628052328 | 7 |
| case/contrast_s2 | common | 8 | 2.80065865176 | 2.80065865176 | 0 | 0 | 0 |
| case/contrast_s2 | covariance | 8 | 2.79802399304 | 2.80065865176 | 0.000940728253586 | 0.554714797202 | 7 |
| case/gamma_s1 | linear | 8 | 1.99219134275 | 3.43798433568 | 0.420535072811 | -0.3923475226 | 3 |
| case/gamma_s1 | common | 8 | 3.43798433568 | 3.43798433568 | 0 | 0 | 0 |
| case/gamma_s1 | covariance | 8 | 3.43782728023 | 3.43798433568 | 4.56824216193e-05 | -0.332652215222 | 3 |
| case/gamma_s2 | linear | 8 | 5.43744202919 | 5.55969607503 | 0.0219893397386 | 0.185307546573 | 4 |
| case/gamma_s2 | common | 8 | 5.55969607503 | 5.55969607503 | 0 | 0 | 0 |
| case/gamma_s2 | covariance | 8 | 5.55290550914 | 5.55969607503 | 0.00122139156485 | -0.166824662056 | 4 |
| family/clean | linear | 32 | 7.54503118743 | 2.09734316375 | -2.59742331052 | -0.0225429079024 | 15 |
| family/clean | common | 32 | 2.09734316375 | 2.09734316375 | 0 | 0 | 0 |
| family/clean | covariance | 32 | 2.09673916223 | 2.09734316375 | 0.000287984117384 | 0.351928061348 | 22 |
| family/color_cast | linear | 8 | 1.58072927522 | 0.124965682548 | -11.6493069376 | -0.127328837355 | 4 |
| family/color_cast | common | 8 | 0.124965682548 | 0.124965682548 | 0 | 0 | 0 |
| family/color_cast | covariance | 8 | 0.123957374357 | 0.124965682548 | 0.00806868069684 | 0.282724513085 | 4 |
| family/contrast | linear | 8 | 2.80194576629 | 2.80065865176 | -0.000459575655253 | 0.380628052328 | 7 |
| family/contrast | common | 8 | 2.80065865176 | 2.80065865176 | 0 | 0 | 0 |
| family/contrast | covariance | 8 | 2.79802399304 | 2.80065865176 | 0.000940728253586 | 0.554714797202 | 7 |
| family/gamma | linear | 16 | 7.42963337194 | 8.99768041071 | 0.174272364342 | -0.155507476852 | 7 |
| family/gamma | common | 16 | 8.99768041071 | 8.99768041071 | 0 | 0 | 0 |
| family/gamma | covariance | 16 | 8.99073278937 | 8.99768041071 | 0.000772156936231 | -0.182313432247 | 7 |

## Full-gradient first-order statistics by scope and reference

| Scope | Method | Median gradient cosine | Negative g dot delta | Mean g dot delta | Median g dot delta | Mean absolute g dot delta |
| --- | --- | --- | --- | --- | --- | --- |
| overall | linear | 0.148616009106 | 37 | -4.64797406584e-05 | -2.44929633149e-06 | 6.10586228282e-05 |
| overall | common | -0.0404019221504 | 29 | -4.11613127456e-06 | 1.97591669459e-07 | 9.58894032334e-06 |
| overall | covariance | 0.13440343542 | 39 | -8.72254564386e-08 | -5.56351938451e-09 | 1.24227771906e-07 |
| clean | linear | 0.0423232374929 | 16 | -3.17178286854e-06 | -1.8654036794e-07 | 2.16322519844e-05 |
| clean | common | -0.089282731244 | 12 | -1.74270176018e-06 | 3.77242793191e-07 | 4.17139153949e-06 |
| clean | covariance | 0.0912012256065 | 19 | 1.97588355994e-09 | -2.81042839866e-09 | 2.92640643096e-08 |
| corrupted | linear | 0.222104104393 | 21 | -8.97876984482e-05 | -4.49167766599e-06 | 0.000100484993672 |
| corrupted | common | 0.0526927579474 | 17 | -6.48956078894e-06 | -3.65507469413e-07 | 1.50064891072e-05 |
| corrupted | covariance | 0.215720038571 | 20 | -1.76426796437e-07 | -2.72189253197e-08 | 2.19191479503e-07 |
| fold0 | linear | 0.00991954289168 | 8 | -5.72228209791e-06 | -4.76342901698e-07 | 1.44892738588e-05 |
| fold0 | common | -0.0404019221504 | 6 | 3.02838350457e-07 | 3.41976451423e-07 | 1.75438988648e-06 |
| fold0 | covariance | -0.163397961683 | 7 | 3.66205486448e-09 | 3.38138112143e-09 | 2.9189536469e-08 |
| fold0/clean | linear | -0.0605722578153 | 3 | 6.97401074271e-07 | 1.44752638373e-06 | 9.52902111188e-06 |
| fold0/clean | common | -0.0631521539927 | 3 | -3.38099329466e-07 | 3.41976451423e-07 | 1.734986615e-06 |
| fold0/clean | covariance | -0.0534537515682 | 4 | 2.38799631906e-09 | 4.60654535417e-10 | 9.45172389589e-09 |
| fold0/corrupted | linear | 0.144460792096 | 5 | -1.21419652701e-05 | -3.36196009991e-06 | 1.94495266057e-05 |
| fold0/corrupted | common | -0.0175271732765 | 3 | 9.4377603038e-07 | 2.89526688647e-07 | 1.77379315796e-06 |
| fold0/corrupted | covariance | -0.180737954892 | 3 | 4.93611340991e-09 | 1.50750638893e-08 | 4.89273490422e-08 |
| fold1 | linear | 0.08623473496 | 8 | -4.10684097379e-05 | -1.05165271086e-06 | 5.0048210265e-05 |
| fold1 | common | 0.0418520835417 | 8 | -4.66159736797e-06 | -8.98150587725e-08 | 1.43350255544e-05 |
| fold1 | covariance | 0.102252140802 | 8 | -1.70369191562e-07 | -2.30552474828e-10 | 2.02897124503e-07 |
| fold1/clean | linear | 0.276895202197 | 5 | -7.05985448447e-06 | -3.12321413822e-06 | 1.4336695611e-05 |
| fold1/clean | common | -0.129673392046 | 3 | -2.7422140717e-06 | 1.61719765533e-07 | 4.62307244444e-06 |
| fold1/clean | covariance | -0.246415368066 | 3 | 2.50613567672e-08 | 2.9920832908e-09 | 3.35118403586e-08 |
| fold1/corrupted | linear | -0.0726907451521 | 3 | -7.50769649913e-05 | 8.08528579042e-07 | 8.5759724919e-05 |
| fold1/corrupted | common | 0.247693072372 | 5 | -6.58098066423e-06 | -3.65507469413e-07 | 2.40469786643e-05 |
| fold1/corrupted | covariance | 0.347479776957 | 5 | -3.65799739891e-07 | -1.45354969333e-08 | 3.72282408648e-07 |
| fold2 | linear | 0.228604268019 | 11 | -0.000111003223772 | -5.5564319555e-06 | 0.000137496788729 |
| fold2 | common | -0.0392248827147 | 7 | -6.27850256895e-06 | 2.32858011227e-07 | 9.83126047774e-06 |
| fold2 | covariance | 0.160863874617 | 13 | -6.64531663052e-08 | -3.88458528896e-08 | 9.86552733197e-08 |
| fold2/clean | linear | 0.0467508602678 | 4 | 1.47929090123e-05 | 8.84833070517e-07 | 3.15892263168e-05 |
| fold2/clean | common | -0.104996873049 | 3 | 1.23180328254e-06 | 8.71533989725e-07 | 3.31558974438e-06 |
| fold2/clean | covariance | 0.144479393216 | 7 | -1.55656456816e-08 | -2.18464162685e-08 | 5.16170012557e-08 |
| fold2/corrupted | linear | 0.456988620546 | 7 | -0.000236799356555 | -2.02340597408e-05 | 0.000243404351141 |
| fold2/corrupted | common | 0.0173131968028 | 4 | -1.37888084204e-05 | -2.70579576351e-07 | 1.63469312111e-05 |
| fold2/corrupted | covariance | 0.276322433299 | 6 | -1.17340686929e-07 | -4.45424579352e-08 | 1.45693545384e-07 |
| fold3 | linear | 0.105157628689 | 10 | -2.81250470261e-05 | -1.68591453045e-06 | 4.220021846e-05 |
| fold3 | common | -0.0396993604103 | 8 | -5.82726351179e-06 | -1.66005988032e-07 | 1.24350853748e-05 |
| fold3 | covariance | 0.215720038571 | 11 | -1.15741522752e-07 | -9.58030620835e-09 | 1.66169153333e-07 |
| fold3/clean | linear | -0.121895678836 | 4 | -2.11175870763e-05 | 1.12817683063e-06 | 3.10740648982e-05 |
| fold3/clean | common | -0.153046260714 | 3 | -5.1222969221e-06 | 6.25609930418e-07 | 7.01191735414e-06 |
| fold3/clean | covariance | 0.111252321634 | 5 | -3.98017316494e-09 | -3.44002809296e-09 | 2.24756917282e-08 |
| fold3/corrupted | linear | 0.174966532179 | 6 | -3.5132506976e-05 | -2.80028996073e-06 | 5.33263720219e-05 |
| fold3/corrupted | common | 0.0845020563871 | 5 | -6.53223010147e-06 | -1.34305646175e-06 | 1.78582533954e-05 |
| fold3/corrupted | covariance | 0.306551170702 | 6 | -2.27502872339e-07 | -5.37576544968e-08 | 3.09862614938e-07 |
| case/clean_s0 | linear | 0.0423232374929 | 16 | -3.17178286854e-06 | -1.8654036794e-07 | 2.16322519844e-05 |
| case/clean_s0 | common | -0.089282731244 | 12 | -1.74270176018e-06 | 3.77242793191e-07 | 4.17139153949e-06 |
| case/clean_s0 | covariance | 0.0912012256065 | 19 | 1.97588355994e-09 | -2.81042839866e-09 | 2.92640643096e-08 |
| case/color_cast_s2 | linear | 0.24319588209 | 7 | -8.32760717037e-06 | -3.36196009991e-06 | 8.92893353478e-06 |
| case/color_cast_s2 | common | 0.175316104798 | 5 | -1.08457617004e-06 | -5.87986372576e-07 | 1.52010615171e-06 |
| case/color_cast_s2 | covariance | 0.0122519060214 | 4 | -1.10619680942e-08 | 1.35905489221e-09 | 4.69016327991e-08 |
| case/contrast_s2 | linear | 0.292468060266 | 6 | -2.5500710265e-05 | -3.78612234263e-05 | 3.91555918718e-05 |
| case/contrast_s2 | common | -0.188538151812 | 1 | 1.29712757495e-05 | 4.35641792868e-06 | 1.31887568408e-05 |
| case/contrast_s2 | covariance | 0.402149779274 | 6 | -1.22319043843e-07 | -9.3274548678e-08 | 1.42715691489e-07 |
| case/gamma_s1 | linear | -0.117387707283 | 3 | -0.000217023789293 | 2.05244362178e-06 | 0.000235850099053 |
| case/gamma_s1 | common | 0.15787488392 | 6 | -1.52888711509e-05 | -1.3406116212e-06 | 2.06695773532e-05 |
| case/gamma_s1 | covariance | -0.0257120716499 | 4 | -6.27704951382e-08 | 2.31412814323e-10 | 1.5154029259e-07 |
| case/gamma_s2 | linear | 0.117064395834 | 5 | -0.000108298687064 | -4.47096692834e-06 | 0.000118005350228 |
| case/gamma_s2 | common | 0.205017221223 | 5 | -2.25560715844e-05 | -1.73000838261e-06 | 2.46475160831e-05 |
| case/gamma_s2 | covariance | 0.318597559217 | 6 | -5.09555678673e-07 | -4.6198640656e-08 | 5.35608301134e-07 |
| family/clean | linear | 0.0423232374929 | 16 | -3.17178286854e-06 | -1.8654036794e-07 | 2.16322519844e-05 |
| family/clean | common | -0.089282731244 | 12 | -1.74270176018e-06 | 3.77242793191e-07 | 4.17139153949e-06 |
| family/clean | covariance | 0.0912012256065 | 19 | 1.97588355994e-09 | -2.81042839866e-09 | 2.92640643096e-08 |
| family/color_cast | linear | 0.24319588209 | 7 | -8.32760717037e-06 | -3.36196009991e-06 | 8.92893353478e-06 |
| family/color_cast | common | 0.175316104798 | 5 | -1.08457617004e-06 | -5.87986372576e-07 | 1.52010615171e-06 |
| family/color_cast | covariance | 0.0122519060214 | 4 | -1.10619680942e-08 | 1.35905489221e-09 | 4.69016327991e-08 |
| family/contrast | linear | 0.292468060266 | 6 | -2.5500710265e-05 | -3.78612234263e-05 | 3.91555918718e-05 |
| family/contrast | common | -0.188538151812 | 1 | 1.29712757495e-05 | 4.35641792868e-06 | 1.31887568408e-05 |
| family/contrast | covariance | 0.402149779274 | 6 | -1.22319043843e-07 | -9.3274548678e-08 | 1.42715691489e-07 |
| family/gamma | linear | -0.0191611490017 | 8 | -0.000162661238179 | -1.78249744444e-06 | 0.000176927724641 |
| family/gamma | common | 0.189320167082 | 11 | -1.89224713676e-05 | -1.63240507549e-06 | 2.26585467181e-05 |
| family/gamma | covariance | 0.20889635004 | 10 | -2.86163086906e-07 | -3.06641889425e-08 | 3.43574296862e-07 |

## Norms by scope and reference

| Scope | Method | Vector | Mean | Median | Max |
| --- | --- | --- | --- | --- | --- |
| overall | linear | true_residual | 0.300986266281 | 0.159951483022 | 1.68066171695 |
| overall | linear | predicted_residual | 0.347331767762 | 0.297071407194 | 1.29687207643 |
| overall | linear | predicted_gradient | 0.353369022239 | 0.300129341593 | 1.20548818379 |
| overall | common | true_residual | 0.300986266281 | 0.159951483022 | 1.68066171695 |
| overall | common | predicted_residual | 0 | 0 | 0 |
| overall | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| overall | covariance | true_residual | 0.300986266281 | 0.159951483022 | 1.68066171695 |
| overall | covariance | predicted_residual | 0.000854726038687 | 0.00083031759374 | 0.00216409607732 |
| overall | covariance | predicted_gradient | 0.000854726038687 | 0.00083031759374 | 0.00216409607732 |
| clean | linear | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| clean | linear | predicted_residual | 0.346130824734 | 0.288886213011 | 1.29687207643 |
| clean | linear | predicted_gradient | 0.35119107452 | 0.296727114266 | 1.20548818379 |
| clean | common | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| clean | common | predicted_residual | 0 | 0 | 0 |
| clean | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| clean | covariance | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| clean | covariance | predicted_residual | 0.000655510576413 | 0.00063407648346 | 0.00187406661567 |
| clean | covariance | predicted_gradient | 0.000655510576413 | 0.00063407648346 | 0.00187406661567 |
| corrupted | linear | true_residual | 0.396971507108 | 0.225550617658 | 1.68066171695 |
| corrupted | linear | predicted_residual | 0.34853271079 | 0.311322233568 | 0.961039236971 |
| corrupted | linear | predicted_gradient | 0.355546969958 | 0.303594634733 | 1.02228446452 |
| corrupted | common | true_residual | 0.396971507108 | 0.225550617658 | 1.68066171695 |
| corrupted | common | predicted_residual | 0 | 0 | 0 |
| corrupted | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| corrupted | covariance | true_residual | 0.396971507108 | 0.225550617658 | 1.68066171695 |
| corrupted | covariance | predicted_residual | 0.00105394150096 | 0.000917925623178 | 0.00216409607732 |
| corrupted | covariance | predicted_gradient | 0.00105394150096 | 0.000917925623178 | 0.00216409607732 |
| fold0 | linear | true_residual | 0.160375728416 | 0.129645878872 | 0.519422512198 |
| fold0 | linear | predicted_residual | 0.428927766383 | 0.384044508553 | 1.29687207643 |
| fold0 | linear | predicted_gradient | 0.416875835489 | 0.31575912287 | 1.20548818379 |
| fold0 | common | true_residual | 0.160375728416 | 0.129645878872 | 0.519422512198 |
| fold0 | common | predicted_residual | 0 | 0 | 0 |
| fold0 | common | predicted_gradient | 0.101733669044 | 0.101733669044 | 0.101733669044 |
| fold0 | covariance | true_residual | 0.160375728416 | 0.129645878872 | 0.519422512198 |
| fold0 | covariance | predicted_residual | 0.00090528556877 | 0.000859317030385 | 0.00216409607732 |
| fold0 | covariance | predicted_gradient | 0.00090528556877 | 0.000859317030385 | 0.00216409607732 |
| fold0/clean | linear | true_residual | 0.129666233978 | 0.129645878872 | 0.164682037369 |
| fold0/clean | linear | predicted_residual | 0.429251050013 | 0.370372246432 | 1.29687207643 |
| fold0/clean | linear | predicted_gradient | 0.406983739034 | 0.314265295461 | 1.20548818379 |
| fold0/clean | common | true_residual | 0.129666233978 | 0.129645878872 | 0.164682037369 |
| fold0/clean | common | predicted_residual | 0 | 0 | 0 |
| fold0/clean | common | predicted_gradient | 0.101733669044 | 0.101733669044 | 0.101733669044 |
| fold0/clean | covariance | true_residual | 0.129666233978 | 0.129645878872 | 0.164682037369 |
| fold0/clean | covariance | predicted_residual | 0.00048402658562 | 0.000516394504489 | 0.0010580926802 |
| fold0/clean | covariance | predicted_gradient | 0.00048402658562 | 0.000516394504489 | 0.0010580926802 |
| fold0/corrupted | linear | true_residual | 0.191085222854 | 0.138366645604 | 0.519422512198 |
| fold0/corrupted | linear | predicted_residual | 0.428604482752 | 0.403538162301 | 0.724720737798 |
| fold0/corrupted | linear | predicted_gradient | 0.426767931944 | 0.331095620008 | 0.701455269842 |
| fold0/corrupted | common | true_residual | 0.191085222854 | 0.138366645604 | 0.519422512198 |
| fold0/corrupted | common | predicted_residual | 0 | 0 | 0 |
| fold0/corrupted | common | predicted_gradient | 0.101733669044 | 0.101733669044 | 0.101733669044 |
| fold0/corrupted | covariance | true_residual | 0.191085222854 | 0.138366645604 | 0.519422512198 |
| fold0/corrupted | covariance | predicted_residual | 0.00132654455192 | 0.00129472513169 | 0.00216409607732 |
| fold0/corrupted | covariance | predicted_gradient | 0.00132654455192 | 0.00129472513169 | 0.00216409607732 |
| fold1 | linear | true_residual | 0.353858977149 | 0.130759185172 | 1.55816195075 |
| fold1 | linear | predicted_residual | 0.292604425129 | 0.272281358311 | 0.627587483926 |
| fold1 | linear | predicted_gradient | 0.312333024582 | 0.281215002889 | 0.670870003982 |
| fold1 | common | true_residual | 0.353858977149 | 0.130759185172 | 1.55816195075 |
| fold1 | common | predicted_residual | 0 | 0 | 0 |
| fold1 | common | predicted_gradient | 0.0743266118063 | 0.0743266118063 | 0.0743266118063 |
| fold1 | covariance | true_residual | 0.353858977149 | 0.130759185172 | 1.55816195075 |
| fold1 | covariance | predicted_residual | 0.000761687795129 | 0.000700793201472 | 0.00165092917075 |
| fold1 | covariance | predicted_gradient | 0.000761687795129 | 0.000700793201472 | 0.00165092917075 |
| fold1/clean | linear | true_residual | 0.205799653729 | 0.130759185172 | 0.523910623416 |
| fold1/clean | linear | predicted_residual | 0.302556462204 | 0.272281358311 | 0.627587483926 |
| fold1/clean | linear | predicted_gradient | 0.318039640962 | 0.276570977945 | 0.670870003982 |
| fold1/clean | common | true_residual | 0.205799653729 | 0.130759185172 | 0.523910623416 |
| fold1/clean | common | predicted_residual | 0 | 0 | 0 |
| fold1/clean | common | predicted_gradient | 0.0743266118063 | 0.0743266118063 | 0.0743266118063 |
| fold1/clean | covariance | true_residual | 0.205799653729 | 0.130759185172 | 0.523910623416 |
| fold1/clean | covariance | predicted_residual | 0.000693355429012 | 0.000705199339576 | 0.00120642642352 |
| fold1/clean | covariance | predicted_gradient | 0.000693355429012 | 0.000705199339576 | 0.00120642642352 |
| fold1/corrupted | linear | true_residual | 0.50191830057 | 0.261648943263 | 1.55816195075 |
| fold1/corrupted | linear | predicted_residual | 0.282652388055 | 0.266549358837 | 0.539464422953 |
| fold1/corrupted | linear | predicted_gradient | 0.306626408203 | 0.281215002889 | 0.571930459491 |
| fold1/corrupted | common | true_residual | 0.50191830057 | 0.261648943263 | 1.55816195075 |
| fold1/corrupted | common | predicted_residual | 0 | 0 | 0 |
| fold1/corrupted | common | predicted_gradient | 0.0743266118063 | 0.0743266118063 | 0.0743266118063 |
| fold1/corrupted | covariance | true_residual | 0.50191830057 | 0.261648943263 | 1.55816195075 |
| fold1/corrupted | covariance | predicted_residual | 0.000830020161247 | 0.000700793201472 | 0.00165092917075 |
| fold1/corrupted | covariance | predicted_gradient | 0.000830020161247 | 0.000700793201472 | 0.00165092917075 |
| fold2 | linear | true_residual | 0.347005658507 | 0.224965410997 | 1.68066171695 |
| fold2 | linear | predicted_residual | 0.421539988927 | 0.351043535145 | 0.961039236971 |
| fold2 | linear | predicted_gradient | 0.426871038991 | 0.337465097403 | 1.02228446452 |
| fold2 | common | true_residual | 0.347005658507 | 0.224965410997 | 1.68066171695 |
| fold2 | common | predicted_residual | 0 | 0 | 0 |
| fold2 | common | predicted_gradient | 0.0686575061037 | 0.0686575061037 | 0.0686575061037 |
| fold2 | covariance | true_residual | 0.347005658507 | 0.224965410997 | 1.68066171695 |
| fold2 | covariance | predicted_residual | 0.000902375263776 | 0.000809875401405 | 0.0019353545073 |
| fold2 | covariance | predicted_gradient | 0.000902375263776 | 0.000809875401405 | 0.0019353545073 |
| fold2/clean | linear | true_residual | 0.269557786889 | 0.214130616045 | 0.629929197293 |
| fold2/clean | linear | predicted_residual | 0.423158433772 | 0.379356304242 | 0.80476779371 |
| fold2/clean | linear | predicted_gradient | 0.433098742535 | 0.383481561297 | 0.865916660135 |
| fold2/clean | common | true_residual | 0.269557786889 | 0.214130616045 | 0.629929197293 |
| fold2/clean | common | predicted_residual | 0 | 0 | 0 |
| fold2/clean | common | predicted_gradient | 0.0686575061037 | 0.0686575061037 | 0.0686575061037 |
| fold2/clean | covariance | true_residual | 0.269557786889 | 0.214130616045 | 0.629929197293 |
| fold2/clean | covariance | predicted_residual | 0.000731157352446 | 0.000617954596097 | 0.00163775271661 |
| fold2/clean | covariance | predicted_gradient | 0.000731157352446 | 0.000617954596097 | 0.00163775271661 |
| fold2/corrupted | linear | true_residual | 0.424453530125 | 0.266362430522 | 1.68066171695 |
| fold2/corrupted | linear | predicted_residual | 0.419921544083 | 0.337189085716 | 0.961039236971 |
| fold2/corrupted | linear | predicted_gradient | 0.420643335447 | 0.306995304386 | 1.02228446452 |
| fold2/corrupted | common | true_residual | 0.424453530125 | 0.266362430522 | 1.68066171695 |
| fold2/corrupted | common | predicted_residual | 0 | 0 | 0 |
| fold2/corrupted | common | predicted_gradient | 0.0686575061037 | 0.0686575061037 | 0.0686575061037 |
| fold2/corrupted | covariance | true_residual | 0.424453530125 | 0.266362430522 | 1.68066171695 |
| fold2/corrupted | covariance | predicted_residual | 0.00107359317511 | 0.000998275243211 | 0.0019353545073 |
| fold2/corrupted | covariance | predicted_gradient | 0.00107359317511 | 0.000998275243211 | 0.0019353545073 |
| fold3 | linear | true_residual | 0.342704701053 | 0.22870067261 | 1.65647510776 |
| fold3 | linear | predicted_residual | 0.246254890608 | 0.246311318312 | 0.493642652909 |
| fold3 | linear | predicted_gradient | 0.257396189894 | 0.255247272399 | 0.533384204197 |
| fold3 | common | true_residual | 0.342704701053 | 0.22870067261 | 1.65647510776 |
| fold3 | common | predicted_residual | 0 | 0 | 0 |
| fold3 | common | predicted_gradient | 0.069041138938 | 0.069041138938 | 0.069041138938 |
| fold3 | covariance | true_residual | 0.342704701053 | 0.22870067261 | 1.65647510776 |
| fold3 | covariance | predicted_residual | 0.000849555527073 | 0.000884374147566 | 0.00187406661567 |
| fold3 | covariance | predicted_gradient | 0.000849555527073 | 0.000884374147566 | 0.00187406661567 |
| fold3/clean | linear | true_residual | 0.214980427224 | 0.14911641581 | 0.493606297813 |
| fold3/clean | linear | predicted_residual | 0.229557352946 | 0.222315306171 | 0.322835237787 |
| fold3/clean | linear | predicted_gradient | 0.246642175547 | 0.245441865215 | 0.356049547778 |
| fold3/clean | common | true_residual | 0.214980427224 | 0.14911641581 | 0.493606297813 |
| fold3/clean | common | predicted_residual | 0 | 0 | 0 |
| fold3/clean | common | predicted_gradient | 0.069041138938 | 0.069041138938 | 0.069041138938 |
| fold3/clean | covariance | true_residual | 0.214980427224 | 0.14911641581 | 0.493606297813 |
| fold3/clean | covariance | predicted_residual | 0.000713502938575 | 0.000607485292809 | 0.00187406661567 |
| fold3/clean | covariance | predicted_gradient | 0.000713502938575 | 0.000607485292809 | 0.00187406661567 |
| fold3/corrupted | linear | true_residual | 0.470428974881 | 0.270016578087 | 1.65647510776 |
| fold3/corrupted | linear | predicted_residual | 0.262952428271 | 0.275733019994 | 0.493642652909 |
| fold3/corrupted | linear | predicted_gradient | 0.26815020424 | 0.264248504667 | 0.533384204197 |
| fold3/corrupted | common | true_residual | 0.470428974881 | 0.270016578087 | 1.65647510776 |
| fold3/corrupted | common | predicted_residual | 0 | 0 | 0 |
| fold3/corrupted | common | predicted_gradient | 0.069041138938 | 0.069041138938 | 0.069041138938 |
| fold3/corrupted | covariance | true_residual | 0.470428974881 | 0.270016578087 | 1.65647510776 |
| fold3/corrupted | covariance | predicted_residual | 0.000985608115571 | 0.000884374147566 | 0.00137690629315 |
| fold3/corrupted | covariance | predicted_gradient | 0.000985608115571 | 0.000884374147566 | 0.00137690629315 |
| case/clean_s0 | linear | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| case/clean_s0 | linear | predicted_residual | 0.346130824734 | 0.288886213011 | 1.29687207643 |
| case/clean_s0 | linear | predicted_gradient | 0.35119107452 | 0.296727114266 | 1.20548818379 |
| case/clean_s0 | common | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| case/clean_s0 | common | predicted_residual | 0 | 0 | 0 |
| case/clean_s0 | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| case/clean_s0 | covariance | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| case/clean_s0 | covariance | predicted_residual | 0.000655510576413 | 0.00063407648346 | 0.00187406661567 |
| case/clean_s0 | covariance | predicted_gradient | 0.000655510576413 | 0.00063407648346 | 0.00187406661567 |
| case/color_cast_s2 | linear | true_residual | 0.111654731553 | 0.0941749647756 | 0.257091421654 |
| case/color_cast_s2 | linear | predicted_residual | 0.382912703921 | 0.34417528725 | 0.724720737798 |
| case/color_cast_s2 | linear | predicted_gradient | 0.395469226033 | 0.360475808403 | 0.701455269842 |
| case/color_cast_s2 | common | true_residual | 0.111654731553 | 0.0941749647756 | 0.257091421654 |
| case/color_cast_s2 | common | predicted_residual | 0 | 0 | 0 |
| case/color_cast_s2 | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| case/color_cast_s2 | covariance | true_residual | 0.111654731553 | 0.0941749647756 | 0.257091421654 |
| case/color_cast_s2 | covariance | predicted_residual | 0.00113033948224 | 0.00105562387714 | 0.0019353545073 |
| case/color_cast_s2 | covariance | predicted_gradient | 0.00113033948224 | 0.00105562387714 | 0.0019353545073 |
| case/contrast_s2 | linear | true_residual | 0.492474133305 | 0.395311498456 | 1.28897012528 |
| case/contrast_s2 | linear | predicted_residual | 0.322840576038 | 0.320642678008 | 0.430404456352 |
| case/contrast_s2 | linear | predicted_gradient | 0.282645560686 | 0.292575736529 | 0.356076186467 |
| case/contrast_s2 | common | true_residual | 0.492474133305 | 0.395311498456 | 1.28897012528 |
| case/contrast_s2 | common | predicted_residual | 0 | 0 | 0 |
| case/contrast_s2 | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| case/contrast_s2 | covariance | true_residual | 0.492474133305 | 0.395311498456 | 1.28897012528 |
| case/contrast_s2 | covariance | predicted_residual | 0.000765999785765 | 0.000833614287356 | 0.001215329556 |
| case/contrast_s2 | covariance | predicted_gradient | 0.000765999785765 | 0.000833614287356 | 0.001215329556 |
| case/gamma_s1 | linear | true_residual | 0.420485905476 | 0.203913320245 | 1.68066171695 |
| case/gamma_s1 | linear | predicted_residual | 0.343046821306 | 0.247320879558 | 0.961039236971 |
| case/gamma_s1 | linear | predicted_gradient | 0.380942522926 | 0.270638165343 | 1.02228446452 |
| case/gamma_s1 | common | true_residual | 0.420485905476 | 0.203913320245 | 1.68066171695 |
| case/gamma_s1 | common | predicted_residual | 0 | 0 | 0 |
| case/gamma_s1 | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| case/gamma_s1 | covariance | true_residual | 0.420485905476 | 0.203913320245 | 1.68066171695 |
| case/gamma_s1 | covariance | predicted_residual | 0.00100866705324 | 0.000883972093027 | 0.00174230373124 |
| case/gamma_s1 | covariance | predicted_gradient | 0.00100866705324 | 0.000883972093027 | 0.00174230373124 |
| case/gamma_s2 | linear | true_residual | 0.563271258096 | 0.227382405849 | 1.65647510776 |
| case/gamma_s2 | linear | predicted_residual | 0.345330741897 | 0.329025631109 | 0.689866443887 |
| case/gamma_s2 | linear | predicted_gradient | 0.363130570189 | 0.285973691358 | 0.742552362748 |
| case/gamma_s2 | common | true_residual | 0.563271258096 | 0.227382405849 | 1.65647510776 |
| case/gamma_s2 | common | predicted_residual | 0 | 0 | 0 |
| case/gamma_s2 | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| case/gamma_s2 | covariance | true_residual | 0.563271258096 | 0.227382405849 | 1.65647510776 |
| case/gamma_s2 | covariance | predicted_residual | 0.00131075968261 | 0.00124942845118 | 0.00216409607732 |
| case/gamma_s2 | covariance | predicted_gradient | 0.00131075968261 | 0.00124942845118 | 0.00216409607732 |
| family/clean | linear | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| family/clean | linear | predicted_residual | 0.346130824734 | 0.288886213011 | 1.29687207643 |
| family/clean | linear | predicted_gradient | 0.35119107452 | 0.296727114266 | 1.20548818379 |
| family/clean | common | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| family/clean | common | predicted_residual | 0 | 0 | 0 |
| family/clean | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| family/clean | covariance | true_residual | 0.205001025455 | 0.141199845431 | 0.629929197293 |
| family/clean | covariance | predicted_residual | 0.000655510576413 | 0.00063407648346 | 0.00187406661567 |
| family/clean | covariance | predicted_gradient | 0.000655510576413 | 0.00063407648346 | 0.00187406661567 |
| family/color_cast | linear | true_residual | 0.111654731553 | 0.0941749647756 | 0.257091421654 |
| family/color_cast | linear | predicted_residual | 0.382912703921 | 0.34417528725 | 0.724720737798 |
| family/color_cast | linear | predicted_gradient | 0.395469226033 | 0.360475808403 | 0.701455269842 |
| family/color_cast | common | true_residual | 0.111654731553 | 0.0941749647756 | 0.257091421654 |
| family/color_cast | common | predicted_residual | 0 | 0 | 0 |
| family/color_cast | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| family/color_cast | covariance | true_residual | 0.111654731553 | 0.0941749647756 | 0.257091421654 |
| family/color_cast | covariance | predicted_residual | 0.00113033948224 | 0.00105562387714 | 0.0019353545073 |
| family/color_cast | covariance | predicted_gradient | 0.00113033948224 | 0.00105562387714 | 0.0019353545073 |
| family/contrast | linear | true_residual | 0.492474133305 | 0.395311498456 | 1.28897012528 |
| family/contrast | linear | predicted_residual | 0.322840576038 | 0.320642678008 | 0.430404456352 |
| family/contrast | linear | predicted_gradient | 0.282645560686 | 0.292575736529 | 0.356076186467 |
| family/contrast | common | true_residual | 0.492474133305 | 0.395311498456 | 1.28897012528 |
| family/contrast | common | predicted_residual | 0 | 0 | 0 |
| family/contrast | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| family/contrast | covariance | true_residual | 0.492474133305 | 0.395311498456 | 1.28897012528 |
| family/contrast | covariance | predicted_residual | 0.000765999785765 | 0.000833614287356 | 0.001215329556 |
| family/contrast | covariance | predicted_gradient | 0.000765999785765 | 0.000833614287356 | 0.001215329556 |
| family/gamma | linear | true_residual | 0.491878581786 | 0.209360418093 | 1.68066171695 |
| family/gamma | linear | predicted_residual | 0.344188781601 | 0.258902092491 | 0.961039236971 |
| family/gamma | linear | predicted_gradient | 0.372036546557 | 0.285973691358 | 1.02228446452 |
| family/gamma | common | true_residual | 0.491878581786 | 0.209360418093 | 1.68066171695 |
| family/gamma | common | predicted_residual | 0 | 0 | 0 |
| family/gamma | common | predicted_gradient | 0.0784397314731 | 0.0716838753722 | 0.101733669044 |
| family/gamma | covariance | true_residual | 0.491878581786 | 0.209360418093 | 1.68066171695 |
| family/gamma | covariance | predicted_residual | 0.00115971336792 | 0.00118083347704 | 0.00216409607732 |
| family/gamma | covariance | predicted_gradient | 0.00115971336792 | 0.00118083347704 | 0.00216409607732 |

## Matched permutation comparisons

| Metric | Observed | Null95 linear | Empirical percentile (<) | Ties | #null>=observed | Corrected upper tail |
| --- | --- | --- | --- | --- | --- | --- |
| R2_residual | -0.380630890016 | -0.545026746385 | 100 | 0 | 0 | 0.0077519379845 |
| median_residual_cosine | 0.0683647559385 | 0.230515231076 | 72.65625 | 0 | 35 | 0.279069767442 |
| positive_residual_dot_count | 33 | 40 | 54.6875 | 14 | 58 | 0.457364341085 |

## All 128 null outcomes

| Permutation | R2_residual | median_residual_cosine | positive_residual_dot_count |
| --- | --- | --- | --- |
| 0 | -0.923370486508 | 0.01464073831 | 32 |
| 1 | -0.979828533209 | -0.220071260963 | 25 |
| 2 | -0.663859102069 | -0.01031940661 | 32 |
| 3 | -1.21001035422 | -0.218813034591 | 24 |
| 4 | -1.25065766907 | 0.0510274279718 | 36 |
| 5 | -0.614076565724 | 0.210796743047 | 38 |
| 6 | -0.541521251365 | 0.214613492768 | 38 |
| 7 | -0.831713548022 | -0.110215057225 | 30 |
| 8 | -0.840490323929 | 0.16514816722 | 38 |
| 9 | -0.901597120053 | -0.26211153468 | 24 |
| 10 | -1.32256188181 | 0.186120489927 | 38 |
| 11 | -0.717147766282 | 0.207072079 | 40 |
| 12 | -0.587249857389 | -0.0237024284298 | 30 |
| 13 | -1.13498019327 | -0.264458245683 | 27 |
| 14 | -1.06836043071 | -0.232232999092 | 26 |
| 15 | -0.940234629793 | 0.247344256756 | 42 |
| 16 | -0.92992331957 | 0.189034508666 | 38 |
| 17 | -1.04557930249 | -0.293863900254 | 26 |
| 18 | -0.788224547749 | 0.0560683508724 | 35 |
| 19 | -1.2282026785 | 0.293527576498 | 39 |
| 20 | -1.28827702725 | -0.0840087078644 | 30 |
| 21 | -1.05947708922 | 0.00161658375058 | 32 |
| 22 | -0.877933976568 | -0.0064454204261 | 32 |
| 23 | -0.955203866869 | -0.151222455377 | 26 |
| 24 | -1.01351313354 | -0.0360932476151 | 31 |
| 25 | -0.907251770575 | 0.0520530016743 | 35 |
| 26 | -0.801728873664 | 0.172490068697 | 40 |
| 27 | -0.407516208574 | -0.247277462467 | 26 |
| 28 | -2.27559514219 | 0.124288829618 | 38 |
| 29 | -0.575508246077 | -0.106897204351 | 25 |
| 30 | -1.1786900803 | 0.148241121728 | 36 |
| 31 | -1.71826281842 | -0.0231367561196 | 29 |
| 32 | -1.81944893368 | 0.283555256104 | 42 |
| 33 | -1.79679778798 | -0.127280051501 | 25 |
| 34 | -0.997948548849 | -0.137153668715 | 27 |
| 35 | -0.961067471264 | 0.0401205948907 | 33 |
| 36 | -0.854664348879 | -0.121960439123 | 29 |
| 37 | -0.903750655679 | 0.0428903355799 | 33 |
| 38 | -0.61200066475 | 0.061817032775 | 33 |
| 39 | -0.571105876264 | -0.000730814316446 | 32 |
| 40 | -1.50508783268 | 0.0301545112363 | 32 |
| 41 | -1.61188785285 | -0.0188054899689 | 32 |
| 42 | -0.74250281378 | -0.0151310295814 | 31 |
| 43 | -0.878721061642 | 0.0164184384204 | 32 |
| 44 | -1.00923620977 | -0.14793839898 | 28 |
| 45 | -0.732582299475 | -0.111842043032 | 27 |
| 46 | -0.858849616321 | 0.0586876138921 | 36 |
| 47 | -0.435762597028 | 0.0729156089106 | 33 |
| 48 | -0.832826433787 | -0.177100688821 | 29 |
| 49 | -0.55885402237 | -0.0473045509516 | 31 |
| 50 | -0.718788420266 | -0.150720305643 | 30 |
| 51 | -1.03249021018 | -0.0742359659478 | 28 |
| 52 | -1.11084932517 | -0.129021369417 | 29 |
| 53 | -1.16096914187 | -0.155277754006 | 28 |
| 54 | -1.22847569284 | 0.0879376363247 | 35 |
| 55 | -0.513310485715 | 0.196348555528 | 41 |
| 56 | -0.630834292108 | 0.219455131047 | 36 |
| 57 | -1.05360474187 | -0.0405252995664 | 30 |
| 58 | -1.6361577252 | 0.0672224289315 | 33 |
| 59 | -1.01914645702 | 0.0299469778425 | 33 |
| 60 | -0.698927151108 | 0.0562250623333 | 36 |
| 61 | -1.03385812905 | 0.038629623194 | 33 |
| 62 | -0.905385003601 | 0.0539743353891 | 33 |
| 63 | -1.09023444338 | 0.248161626296 | 41 |
| 64 | -0.732047328562 | -0.0938143182553 | 30 |
| 65 | -0.844033403035 | 0.0731058658469 | 35 |
| 66 | -0.604635147186 | 0.26793758057 | 39 |
| 67 | -0.700449546086 | 0.168101841841 | 38 |
| 68 | -0.636158755408 | 0.0642137034753 | 37 |
| 69 | -0.813244789062 | -0.116349859749 | 28 |
| 70 | -0.635161346139 | 0.0245058964758 | 33 |
| 71 | -0.929366453297 | 0.158195278667 | 35 |
| 72 | -0.624135514648 | -0.119649709526 | 26 |
| 73 | -1.06385975006 | -0.221076464194 | 26 |
| 74 | -0.654430666504 | -0.017488582599 | 30 |
| 75 | -1.3161163435 | -0.272209083527 | 20 |
| 76 | -0.93178658386 | -0.333632449442 | 22 |
| 77 | -0.713426487 | 0.0468735098508 | 33 |
| 78 | -0.927222108971 | -0.0408174391046 | 32 |
| 79 | -0.518053056008 | -0.0591400531136 | 30 |
| 80 | -0.968301425644 | -0.0669096029455 | 30 |
| 81 | -0.57265949017 | -0.00719821631934 | 31 |
| 82 | -0.848449378708 | 0.0356164148931 | 35 |
| 83 | -0.754804579478 | -0.0311555474732 | 31 |
| 84 | -0.589155869362 | 0.000142813677682 | 32 |
| 85 | -0.989401538207 | 0.197597293325 | 37 |
| 86 | -0.848673321763 | -0.00347259495871 | 32 |
| 87 | -1.72383322758 | 0.134157739113 | 33 |
| 88 | -1.12220576971 | 0.198459299303 | 37 |
| 89 | -0.743513715107 | -0.0079588695932 | 31 |
| 90 | -0.697466311625 | -0.160944954061 | 29 |
| 91 | -0.84564408509 | 0.0366799623571 | 35 |
| 92 | -1.20281939201 | -0.00303988682383 | 32 |
| 93 | -0.794952754424 | 0.20756966402 | 35 |
| 94 | -0.644161341167 | -0.160305682034 | 28 |
| 95 | -0.791478096865 | 0.17518407685 | 38 |
| 96 | -0.855863075992 | -0.0502379767887 | 29 |
| 97 | -0.749837076173 | 0.0907433080891 | 37 |
| 98 | -0.65267681951 | -0.121186628525 | 27 |
| 99 | -0.801846365536 | -0.106904238336 | 30 |
| 100 | -1.18878207669 | -0.135649129488 | 28 |
| 101 | -1.41663702039 | 0.0635346024939 | 34 |
| 102 | -0.948654825259 | -0.108134312591 | 31 |
| 103 | -0.532848596806 | 0.0613571327954 | 34 |
| 104 | -1.03762317075 | -0.0620394484782 | 31 |
| 105 | -0.586612043633 | 0.132685280347 | 35 |
| 106 | -0.620350201762 | -0.00797009067398 | 32 |
| 107 | -0.551536951423 | -0.115641696764 | 28 |
| 108 | -0.688006102786 | 0.181977406928 | 39 |
| 109 | -0.499763001911 | 0.00179719840728 | 32 |
| 110 | -0.825184688035 | -0.000702360498115 | 32 |
| 111 | -0.845764525791 | -0.0500003169431 | 31 |
| 112 | -0.916648391438 | 0.226376145534 | 37 |
| 113 | -1.48163221098 | 0.0920102091199 | 33 |
| 114 | -0.65353130258 | -0.0151786254661 | 32 |
| 115 | -1.54243460827 | 0.068230120995 | 33 |
| 116 | -0.995351198486 | -0.0161098443008 | 32 |
| 117 | -0.809108292976 | 0.0315940643928 | 33 |
| 118 | -1.35583940047 | 0.0502726441768 | 34 |
| 119 | -0.844427461019 | -0.00991616874496 | 31 |
| 120 | -1.3286399742 | -0.118254976419 | 27 |
| 121 | -0.573488669728 | 0.124714460312 | 36 |
| 122 | -1.32245193905 | 0.232743969445 | 40 |
| 123 | -0.798009458127 | -0.252451936424 | 26 |
| 124 | -0.594728638861 | 0.0693938597738 | 34 |
| 125 | -0.977765549747 | 0.271276506075 | 40 |
| 126 | -0.751694646603 | -0.102605583452 | 28 |
| 127 | -0.736047336964 | 0.0220150424014 | 34 |

## All held-out episode outcomes

| Episode | Image | Case | Fold | Method | SSE | Residual dot | Residual cosine | Gradient cosine | g dot delta | norm r | norm rhat | norm ghat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 182164 | clean_s0 | 0 | linear | 0.0349771421016 | -0.000122571820609 | -0.00787134620679 | -0.914562756216 | 8.3477530371e-06 | 0.158282188215 | 0.0983806248923 | 0.0981448867923 |
| 0 | 182164 | clean_s0 | 0 | common | 0.0250532511061 | 0 | 0 | -0.319948762306 | 3.02714723252e-06 | 0.158282188215 | 0 | 0.101733669044 |
| 0 | 182164 | clean_s0 | 0 | covariance | 0.024957203886 | 4.81216788238e-05 | 0.686480800727 | 0.293929516832 | -1.21063139449e-08 | 0.158282188215 | 0.000442874147132 | 0.000442874147132 |
| 1 | 182164 | gamma_s1 | 0 | linear | 0.0819299023871 | -0.0171584967595 | -0.720894605113 | -0.590755061174 | 1.43357409451e-05 | 0.155832050158 | 0.152739258253 | 0.193892915523 |
| 1 | 182164 | gamma_s1 | 0 | common | 0.0242836278563 | 0 | 0 | 0.0679388296255 | -8.65034683996e-07 | 0.155832050158 | 0 | 0.101733669044 |
| 1 | 182164 | gamma_s1 | 0 | covariance | 0.0244729887024 | -9.4323985612e-05 | -0.71690022898 | -0.222136110222 | 2.34734300256e-08 | 0.155832050158 | 0.000844319150543 | 0.000844319150543 |
| 2 | 522454 | clean_s0 | 0 | linear | 1.42946181403 | 0.13622605871 | 0.742077775986 | 0.15179570739 | -1.54209436626e-05 | 0.141551223346 | 1.29687207643 | 1.20548818379 |
| 2 | 522454 | clean_s0 | 0 | common | 0.0200367488307 | 0 | 0 | -0.15076099224 | 1.29253465837e-06 | 0.141551223346 | 0 | 0.101733669044 |
| 2 | 522454 | clean_s0 | 0 | covariance | 0.0198950532062 | 7.14075923285e-05 | 0.476767950108 | -0.187456123803 | 1.6715210296e-08 | 0.141551223346 | 0.0010580926802 | 0.0010580926802 |
| 3 | 522454 | gamma_s2 | 0 | linear | 0.0774646734669 | 0.0395170664755 | 0.867742267832 | 0.451671743574 | -1.17031831992e-05 | 0.12090124105 | 0.376671868249 | 0.30611505355 |
| 3 | 522454 | gamma_s2 | 0 | common | 0.0146171100875 | 0 | 0 | 0.16822628422 | -1.44862159258e-06 | 0.12090124105 | 0 | 0.101733669044 |
| 3 | 522454 | gamma_s2 | 0 | covariance | 0.0147233094271 | -5.231842447e-05 | -0.346190539837 | 0.479430099639 | -5.07259737893e-08 | 0.12090124105 | 0.00124999622713 | 0.00124999622713 |
| 4 | 425480 | clean_s0 | 0 | linear | 0.113488698673 | -0.0169150251661 | -0.52906666645 | -0.0622549091507 | 1.40288236514e-06 | 0.126792707598 | 0.252155225289 | 0.32540319219 |
| 4 | 425480 | clean_s0 | 0 | common | 0.0160763907001 | 0 | 0 | -0.0660705712482 | 4.65477413798e-07 | 0.126792707598 | 0 | 0.101733669044 |
| 4 | 425480 | clean_s0 | 0 | covariance | 0.0159602368834 | 5.82879616532e-05 | 0.707576665226 | 0.233990486082 | -1.05277395651e-08 | 0.126792707598 | 0.000649697341936 | 0.000649697341936 |
| 5 | 425480 | contrast_s2 | 0 | linear | 0.0446440599359 | 0.100484856912 | 0.950229390617 | 0.86030392868 | -6.07135562787e-05 | 0.245694480428 | 0.430404456352 | 0.356076186467 |
| 5 | 425480 | contrast_s2 | 0 | common | 0.0603657777128 | 0 | 0 | -0.266208070252 | 5.36755818562e-06 | 0.245694480428 | 0 | 0.101733669044 |
| 5 | 425480 | contrast_s2 | 0 | covariance | 0.0599768277772 | 0.000194891046139 | 0.869548760072 | 0.683276767745 | -1.23535004984e-07 | 0.245694480428 | 0.000912226210845 | 0.000912226210845 |
| 6 | 410054 | clean_s0 | 0 | linear | 0.169952906332 | 0.0262896755955 | 0.542992431951 | -0.0588896064799 | 1.49217040232e-06 | 0.105291101999 | 0.459832623204 | 0.403770266467 |
| 6 | 410054 | clean_s0 | 0 | common | 0.0110862161602 | 0 | 0 | 0.250746666061 | -1.60083035093e-06 | 0.105291101999 | 0 | 0.101733669044 |
| 6 | 410054 | clean_s0 | 0 | covariance | 0.0110581647841 | 1.40504429543e-05 | 0.599725733864 | -0.342062417238 | 4.7763534319e-09 | 0.105291101999 | 0.000222508015489 | 0.000222508015489 |
| 7 | 410054 | color_cast_s2 | 0 | linear | 0.413456510685 | 0.0605630360557 | 0.863659474831 | 0.210192891928 | -4.36835203128e-06 | 0.0967596765266 | 0.724720737798 | 0.648974340553 |
| 7 | 410054 | color_cast_s2 | 0 | common | 0.00936243500153 | 0 | 0 | 0.30891508858 | -1.00641223375e-06 | 0.0967596765266 | 0 | 0.101733669044 |
| 7 | 410054 | color_cast_s2 | 0 | covariance | 0.00914375311468 | 0.000110444515552 | 0.768306288362 | -0.351725764647 | 1.6733686165e-08 | 0.0967596765266 | 0.00148564607181 | 0.00148564607181 |
| 8 | 256231 | clean_s0 | 0 | linear | 0.183192360456 | 0.00842030191983 | 0.17686533775 | 0.497953492002 | -7.89406282591e-06 | 0.109808193021 | 0.433560981917 | 0.444645181295 |
| 8 | 256231 | clean_s0 | 0 | common | 0.0120578392545 | 0 | 0 | -0.0602337367371 | 2.18475489047e-07 | 0.109808193021 | 0 | 0.101733669044 |
| 8 | 256231 | clean_s0 | 0 | covariance | 0.0121529245375 | -4.73059656628e-05 | -0.626165652179 | 0.185725024523 | -4.55575705719e-09 | 0.109808193021 | 0.000688005525603 | 0.000688005525603 |
| 9 | 256231 | gamma_s1 | 0 | linear | 0.462084798663 | -0.0589060615939 | -0.91140136966 | -0.153010198755 | 3.66516817086e-06 | 0.112225595232 | 0.575915003494 | 0.673416068543 |
| 9 | 256231 | gamma_s1 | 0 | common | 0.0125945842251 | 0 | 0 | -0.135349565158 | 4.89791929346e-07 | 0.112225595232 | 0 | 0.101733669044 |
| 9 | 256231 | gamma_s1 | 0 | covariance | 0.0128932970235 | -0.000147838588032 | -0.756087436778 | 0.0274945384726 | -1.70396375545e-09 | 0.112225595232 | 0.00174230373124 | 0.00174230373124 |
| 10 | 370279 | clean_s0 | 0 | linear | 0.101661719761 | 0.0345508314735 | 0.66620193405 | -0.0793257136672 | 1.86853486181e-06 | 0.132499050145 | 0.391417148856 | 0.303127398732 |
| 10 | 370279 | clean_s0 | 0 | common | 0.0175559982893 | 0 | 0 | -0.0738622299463 | 5.83914348391e-07 | 0.132499050145 | 0 | 0.101733669044 |
| 10 | 370279 | clean_s0 | 0 | covariance | 0.0175766837781 | -1.0328266345e-05 | -0.458083361434 | 0.0805486206665 | -1.06509974013e-09 | 0.132499050145 | 0.000170164964119 | 0.000170164964119 |
| 11 | 370279 | gamma_s2 | 0 | linear | 0.0859100159022 | -0.0127781884601 | -0.424766063466 | -0.504987562821 | 1.12293362267e-05 | 0.166726245855 | 0.180432807231 | 0.233139405157 |
| 11 | 370279 | gamma_s2 | 0 | common | 0.0277976410568 | 0 | 0 | -0.430290898679 | 4.17527021379e-06 | 0.166726245855 | 0 | 0.101733669044 |
| 11 | 370279 | gamma_s2 | 0 | covariance | 0.0283622603464 | -0.000279967988874 | -0.775939513899 | -0.484253278666 | 9.99555301704e-08 | 0.166726245855 | 0.00216409607732 | 0.00216409607732 |
| 12 | 399865 | clean_s0 | 0 | linear | 0.153431510156 | -0.00214087172653 | -0.037214469598 | -0.656108374362 | 2.77943480783e-05 | 0.164682037369 | 0.349327344008 | 0.27090649707 |
| 12 | 399865 | clean_s0 | 0 | common | 0.0271201734319 | 0 | 0 | 0.241446506667 | -3.84102491529e-06 | 0.164682037369 | 0 | 0.101733669044 |
| 12 | 399865 | clean_s0 | 0 | covariance | 0.0270788841368 | 2.08186473477e-05 | 0.214297418693 | -0.258880902198 | 2.38809083209e-08 | 0.164682037369 | 0.000589914861846 | 0.000589914861846 |
| 13 | 399865 | contrast_s2 | 0 | linear | 0.254148284645 | 0.0752097779974 | 0.394420967876 | 0.308838019934 | -4.72253078257e-05 | 0.519422512198 | 0.367107742308 | 0.301074215916 |
| 13 | 399865 | contrast_s2 | 0 | common | 0.269799746178 | 0 | 0 | -0.0144842389892 | 7.48394976671e-07 | 0.519422512198 | 0 | 0.101733669044 |
| 13 | 399865 | contrast_s2 | 0 | covariance | 0.2697869246 | 6.79300232495e-06 | 0.0149579855764 | -0.139339799563 | 6.18747618339e-08 | 0.519422512198 | 0.000874314910227 | 0.000874314910227 |
| 14 | 54355 | clean_s0 | 0 | linear | 0.022788394584 | 0.00507177057509 | 0.337985980099 | 0.827947004928 | -1.2011473662e-05 | 0.0984233701307 | 0.152462375511 | 0.204384305939 |
| 14 | 54355 | clean_s0 | 0 | common | 0.00968715978789 | 0 | 0 | 0.39473749247 | -2.85048851164e-06 | 0.0984233701307 | 0 | 0.101733669044 |
| 14 | 54355 | clean_s0 | 0 | covariance | 0.00968255677427 | 2.30280502393e-06 | 0.459167206925 | -0.54920483357 | 1.98640881096e-09 | 0.0984233701307 | 5.09551486364e-05 | 5.09551486364e-05 |
| 15 | 54355 | color_cast_s2 | 0 | linear | 0.489147747203 | -0.0456764195442 | -0.662090586493 | 0.0787286922633 | -2.35556816854e-06 | 0.111119981388 | 0.620843988334 | 0.701455269842 |
| 15 | 54355 | color_cast_s2 | 0 | common | 0.0123476502636 | 0 | 0 | -0.0205701075638 | 8.92614479476e-08 | 0.111119981388 | 0 | 0.101733669044 |
| 15 | 54355 | color_cast_s2 | 0 | covariance | 0.0121474210217 | 0.000101011689485 | 0.678659073349 | -0.234826374302 | 1.34164416136e-08 | 0.111119981388 | 0.00133945403624 | 0.00133945403624 |
| 16 | 496294 | clean_s0 | 1 | linear | 0.199069276072 | 0.0361871667786 | 0.338733361448 | 0.350681036603 | -2.78882223447e-05 | 0.468436873379 | 0.228058118223 | 0.166921355224 |
| 16 | 496294 | clean_s0 | 1 | common | 0.219433104341 | 0 | 0 | 0.184607025038 | -6.53716140626e-06 | 0.468436873379 | 0 | 0.0743266118063 |
| 16 | 496294 | clean_s0 | 1 | covariance | 0.219420741894 | 6.18148164646e-06 | 0.580799855078 | 0.669637787627 | -7.24855790599e-09 | 0.468436873379 | 2.27203496831e-05 | 2.27203496831e-05 |
| 17 | 496294 | gamma_s1 | 1 | linear | 0.282831803127 | -0.0368251126171 | -0.497603441011 | -0.284880818367 | 1.86724814355e-05 | 0.422492047559 | 0.175162917428 | 0.137716019142 |
| 17 | 496294 | gamma_s1 | 1 | common | 0.178499530251 | 0 | 0 | 0.75683074585 | -2.67730266498e-05 | 0.422492047559 | 0 | 0.0743266118063 |
| 17 | 496294 | gamma_s1 | 1 | covariance | 0.178051596169 | 0.000224218258903 | 0.748707423554 | 0.810403737324 | -2.73398483162e-07 | 0.422492047559 | 0.000708827074918 | 0.000708827074918 |
| 18 | 364188 | clean_s0 | 1 | linear | 0.0751001824393 | -0.00390351728812 | -0.11801177585 | -0.586115189869 | 1.36685106625e-05 | 0.165782190272 | 0.199522964222 | 0.170375379255 |
| 18 | 364188 | clean_s0 | 1 | common | 0.0274837346113 | 0 | 0 | -0.158443926137 | 1.61194906537e-06 | 0.165782190272 | 0 | 0.0743266118063 |
| 18 | 364188 | clean_s0 | 1 | covariance | 0.0274155703437 | 3.41082115157e-05 | 0.900888917837 | 0.637709060335 | -1.99344290194e-08 | 0.165782190272 | 0.000228375667568 | 0.000228375667568 |
| 19 | 364188 | gamma_s2 | 1 | linear | 0.0493533245906 | -0.0090405198268 | -0.611868277213 | -0.0636162744928 | 1.17733808538e-06 | 0.102564481511 | 0.144058363414 | 0.194886711783 |
| 19 | 364188 | gamma_s2 | 1 | common | 0.0105194728677 | 0 | 0 | 0.284972094799 | -2.01139517263e-06 | 0.102564481511 | 0 | 0.0743266118063 |
| 19 | 364188 | gamma_s2 | 1 | covariance | 0.0106030283816 | -4.12467271521e-05 | -0.390227663567 | 0.283422963376 | -2.77370467565e-08 | 0.102564481511 | 0.00103056277509 | 0.00103056277509 |
| 20 | 83968 | clean_s0 | 1 | linear | 0.239965269007 | 0.0200530006326 | 0.511985744226 | 0.592766650456 | -4.72677353027e-05 | 0.523910623416 | 0.0747591395401 | 0.142243097254 |
| 20 | 83968 | clean_s0 | 1 | common | 0.274482341328 | 0 | 0 | 0.5437101641 | -2.26548723955e-05 | 0.523910623416 | 0 | 0.0743266118063 |
| 20 | 83968 | clean_s0 | 1 | covariance | 0.274702895507 | -0.000109975791005 | -0.27041212811 | -0.377448192502 | 1.64255872795e-07 | 0.523910623416 | 0.000776271646116 | 0.000776271646116 |
| 21 | 83968 | contrast_s2 | 1 | linear | 1.59924667486 | 0.0505842094601 | 0.198792835351 | -0.0861880616661 | 2.00361956596e-05 | 1.28897012528 | 0.197411017685 | 0.188329130509 |
| 21 | 83968 | contrast_s2 | 1 | common | 1.66144398387 | 0 | 0 | -0.720525842587 | 6.61065582558e-05 | 1.28897012528 | 0 | 0.0743266118063 |
| 21 | 83968 | contrast_s2 | 1 | covariance | 1.66036072821 | 0.00054174889657 | 0.85414507146 | 0.837494950012 | -5.08693736674e-07 | 1.28897012528 | 0.000492066211332 | 0.000492066211332 |
| 22 | 92109 | clean_s0 | 1 | linear | 0.44512286971 | -0.0227055268798 | -0.47319113305 | 0.31770465998 | -2.54302449443e-06 | 0.0764576089703 | 0.627587483926 | 0.670870003982 |
| 22 | 92109 | clean_s0 | 1 | common | 0.00584576596946 | 0 | 0 | -0.100902857955 | 8.94821452666e-08 | 0.0764576089703 | 0 | 0.0743266118063 |
| 22 | 92109 | clean_s0 | 1 | covariance | 0.0059343213696 | -4.40767056256e-05 | -0.909245985643 | -0.115382543631 | 8.72842160336e-10 | 0.0764576089703 | 0.000634025933884 | 0.000634025933884 |
| 23 | 92109 | color_cast_s2 | 1 | linear | 0.317986643066 | -0.0111221732614 | -0.300079227182 | 0.761787706814 | -4.61500330071e-06 | 0.0687054067102 | 0.539464422953 | 0.571930459491 |
| 23 | 92109 | color_cast_s2 | 1 | common | 0.00472043291122 | 0 | 0 | 0.581870809504 | -4.58105742578e-07 | 0.0687054067102 | 0 | 0.0743266118063 |
| 23 | 92109 | color_cast_s2 | 1 | covariance | 0.00472523940658 | -2.35642668234e-06 | -0.112079888716 | 0.411536590537 | -1.33394710999e-09 | 0.0687054067102 | 0.000306009793505 | 0.000306009793505 |
| 24 | 308546 | clean_s0 | 1 | linear | 0.111936954662 | -0.0114768190498 | -0.508427796376 | -0.423572464533 | 1.75866581116e-06 | 0.0784321115933 | 0.287805004183 | 0.329029462062 |
| 24 | 308546 | clean_s0 | 1 | common | 0.00615159612899 | 0 | 0 | -0.249443166275 | 2.339573858e-07 | 0.0784321115933 | 0 | 0.0743266118063 |
| 24 | 308546 | clean_s0 | 1 | covariance | 0.00630451754388 | -7.59480181164e-05 | -0.956269711797 | -0.400009675284 | 5.11132442126e-09 | 0.0784321115933 | 0.00101260982614 | 0.00101260982614 |
| 25 | 308546 | gamma_s1 | 1 | linear | 0.0961602430402 | -0.0166380169764 | -0.953762927844 | -0.0817652158113 | 4.39719072705e-07 | 0.072685173683 | 0.240002238769 | 0.308182246814 |
| 25 | 308546 | gamma_s1 | 1 | common | 0.00528313447332 | 0 | 0 | 0.210414049945 | -2.72909196248e-07 | 0.072685173683 | 0 | 0.0743266118063 |
| 25 | 308546 | gamma_s1 | 1 | covariance | 0.00550872618754 | -0.000111558071127 | -0.975477791262 | -0.0789186817724 | 2.1667893841e-09 | 0.072685173683 | 0.00157339504378 | 0.00157339504378 |
| 26 | 18507 | clean_s0 | 1 | linear | 0.136125664178 | -0.0251814252402 | -0.696313288334 | -0.534498489959 | 1.36801880324e-05 | 0.140848467517 | 0.256757712438 | 0.319946777958 |
| 26 | 18507 | clean_s0 | 1 | common | 0.0198382908018 | 0 | 0 | -0.665550799503 | 3.95724909581e-06 | 0.140848467517 | 0 | 0.0743266118063 |
| 26 | 18507 | clean_s0 | 1 | covariance | 0.0199811468951 | -7.12269881198e-05 | -0.797473364412 | -0.52861559142 | 2.68153876319e-08 | 0.140848467517 | 0.000634127033036 | 0.000634127033036 |
| 27 | 18507 | gamma_s2 | 1 | linear | 1.53587321929 | 0.498916932615 | 0.984224054907 | 0.98214624053 | -0.000610234617313 | 1.55816195075 | 0.32532817239 | 0.384714742718 |
| 27 | 18507 | gamma_s2 | 1 | common | 2.42786866477 | 0 | 0 | 0.774711407332 | -9.29964005529e-05 | 1.55816195075 | 0 | 0.0743266118063 |
| 27 | 18507 | gamma_s2 | 1 | covariance | 2.42382005429 | 0.00202566802614 | 0.787457705136 | 0.80304506742 | -2.14116538045e-06 | 1.55816195075 | 0.00165092917075 | 0.00165092917075 |
| 28 | 273068 | clean_s0 | 1 | linear | 0.0517875507229 | 0.0234273902269 | 0.669538787394 | 0.236085744413 | -4.18381445806e-06 | 0.120669902828 | 0.28996742184 | 0.233195177932 |
| 28 | 273068 | clean_s0 | 1 | common | 0.0145612254484 | 0 | 0 | -0.288716870263 | 1.63079579869e-06 | 0.120669902828 | 0 | 0.0743266118063 |
| 28 | 273068 | clean_s0 | 1 | covariance | 0.0148065025202 | -0.000121910803548 | -0.837418189904 | -0.406157963991 | 3.7237361495e-08 | 0.120669902828 | 0.00120642642352 | 0.00120642642352 |
| 29 | 273068 | contrast_s2 | 1 | linear | 0.172449242413 | 0.0452364508143 | 0.36683513678 | 0.276098100599 | -2.8497139027e-05 | 0.420733405015 | 0.293096478905 | 0.254247758963 |
| 29 | 273068 | contrast_s2 | 1 | common | 0.177016598095 | 0 | 0 | -0.110868233373 | 3.34527767174e-06 | 0.420733405015 | 0 | 0.0743266118063 |
| 29 | 273068 | contrast_s2 | 1 | covariance | 0.177070890499 | -2.71289759538e-05 | -0.347392634734 | -0.261601608387 | 1.97118287488e-08 | 0.420733405015 | 0.00018561189258 | 0.00018561189258 |
| 30 | 156876 | clean_s0 | 1 | linear | 0.254644357329 | -0.0207750911493 | -0.63401575397 | 0.543103350534 | -3.70340378202e-06 | 0.0718594518592 | 0.455993853258 | 0.511735874029 |
| 30 | 156876 | clean_s0 | 1 | common | 0.0051637808215 | 0 | 0 | 0.271716762715 | -2.69112262812e-07 | 0.0718594518592 | 0 | 0.0743266118063 |
| 30 | 156876 | clean_s0 | 1 | covariance | 0.00529399883776 | -6.45762003649e-05 | -0.870539171806 | 0.481189278638 | -6.61894744013e-09 | 0.0718594518592 | 0.00103228655215 | 0.00103228655215 |
| 31 | 156876 | color_cast_s2 | 1 | linear | 0.175600798645 | -0.0244182774166 | -0.869161633004 | -0.394698034441 | 2.40530545765e-06 | 0.081033814043 | 0.346695492895 | 0.413004196201 |
| 31 | 156876 | color_cast_s2 | 1 | common | 0.00656647901835 | 0 | 0 | -0.375808465123 | 4.12156072687e-07 | 0.081033814043 | 0 | 0.0743266118063 |
| 31 | 156876 | color_cast_s2 | 1 | covariance | 0.00666072149602 | -4.68812810883e-05 | -0.835123698738 | -0.396407938823 | 4.05205689441e-09 | 0.081033814043 | 0.000692759328027 | 0.000692759328027 |
| 32 | 469614 | clean_s0 | 2 | linear | 0.0239459404046 | -0.00254592556919 | -0.279111258251 | 0.145436310822 | -1.92208656748e-06 | 0.108662360928 | 0.0839439133217 | 0.134854485035 |
| 32 | 469614 | clean_s0 | 2 | common | 0.0118075086825 | 0 | 0 | 0.186572047671 | -1.25536293366e-06 | 0.108662360928 | 0 | 0.0686575061037 |
| 32 | 469614 | clean_s0 | 2 | covariance | 0.0117514839588 | 2.80913170196e-05 | 0.650560516981 | 0.134437134138 | -5.23551257454e-09 | 0.108662360928 | 0.000397379319806 | 0.000397379319806 |
| 33 | 469614 | gamma_s1 | 2 | linear | 0.070736394356 | 0.00463482098921 | 0.169860800273 | 0.616755470322 | -2.32976677873e-05 | 0.103684114608 | 0.263164664635 | 0.321794984042 |
| 33 | 469614 | gamma_s1 | 2 | common | 0.0107503956221 | 0 | 0 | 0.480375584914 | -3.87158784925e-06 | 0.103684114608 | 0 | 0.0686575061037 |
| 33 | 469614 | gamma_s1 | 2 | covariance | 0.010753020435 | -1.13586390763e-06 | -0.0184363246568 | 0.481578752185 | -3.35913311285e-08 | 0.103684114608 | 0.000594209689301 | 0.000594209689301 |
| 34 | 298468 | clean_s0 | 2 | linear | 1.13276204952 | -0.0821985047044 | -0.172601056976 | -0.0902454164215 | 4.53981121472e-05 | 0.629929197293 | 0.756012067701 | 0.810240680706 |
| 34 | 298468 | clean_s0 | 2 | common | 0.396810793602 | 0 | 0 | -0.0776626045331 | 3.31053795156e-06 | 0.629929197293 | 0 | 0.0686575061037 |
| 34 | 298468 | clean_s0 | 2 | covariance | 0.396642496953 | 8.44998906801e-05 | 0.159972698159 | 0.0691880954426 | -3.60203982566e-08 | 0.629929197293 | 0.000838529872387 | 0.000838529872387 |
| 35 | 298468 | gamma_s2 | 2 | linear | 0.830523034388 | -0.0594665451311 | -0.17756265703 | -0.0747935232887 | 2.64199783441e-05 | 0.485462906642 | 0.689866443887 | 0.742552362748 |
| 35 | 298468 | gamma_s2 | 2 | common | 0.235674233726 | 0 | 0 | -0.0713590380184 | 2.33065715197e-06 | 0.485462906642 | 0 | 0.0686575061037 |
| 35 | 298468 | gamma_s2 | 2 | covariance | 0.235666720563 | 3.96908107837e-06 | 0.0125412157251 | 0.134369736703 | -4.16713075226e-08 | 0.485462906642 | 0.000651919963327 | 0.000651919963327 |
| 36 | 365614 | clean_s0 | 2 | linear | 0.287301733744 | 0.00969395615585 | 0.0913079594051 | -0.0519345902861 | 3.69175270852e-06 | 0.513802644678 | 0.206631286058 | 0.13926862639 |
| 36 | 365614 | clean_s0 | 2 | common | 0.263993157678 | 0 | 0 | 0.0177374474928 | -6.21586748508e-07 | 0.513802644678 | 0 | 0.0686575061037 |
| 36 | 365614 | clean_s0 | 2 | covariance | 0.263547093009 | 0.000224373451741 | 0.266640923792 | 0.154521652295 | -1.29169587329e-07 | 0.513802644678 | 0.00163775271661 | 0.00163775271661 |
| 37 | 365614 | contrast_s2 | 2 | linear | 0.075862169707 | 0.0840391416818 | 0.743940428354 | 0.62889644601 | -5.41858480384e-05 | 0.27563343939 | 0.409837358181 | 0.353135210765 |
| 37 | 365614 | contrast_s2 | 2 | common | 0.0759737929099 | 0 | 0 | -0.350134320706 | 5.86527553174e-06 | 0.27563343939 | 0 | 0.0686575061037 |
| 37 | 365614 | contrast_s2 | 2 | covariance | 0.0754464215028 | 0.000264424216509 | 0.789360267032 | 0.654070257821 | -1.93947493965e-07 | 0.27563343939 | 0.001215329556 | 0.001215329556 |
| 38 | 117105 | clean_s0 | 2 | linear | 0.0937101799926 | 0.0412432934144 | 0.53187221473 | 0.246447869019 | -1.33270274708e-05 | 0.215140764046 | 0.360431988685 | 0.294269761262 |
| 38 | 117105 | clean_s0 | 2 | common | 0.0462855483544 | 0 | 0 | -0.309195977669 | 3.90107633457e-06 | 0.215140764046 | 0 | 0.0686575061037 |
| 38 | 117105 | clean_s0 | 2 | covariance | 0.0460724719996 | 0.000107100689196 | 0.469341269824 | 0.235330878865 | -4.58693219531e-08 | 0.215140764046 | 0.0010606712931 | 0.0010606712931 |
| 39 | 117105 | color_cast_s2 | 2 | linear | 0.050062684893 | 0.0663807544908 | 0.755730089286 | 0.669995444813 | -4.90667390839e-05 | 0.257091421654 | 0.341655081605 | 0.292195624731 |
| 39 | 117105 | color_cast_s2 | 2 | common | 0.0660959990881 | 0 | 0 | 0.0417171210165 | -7.17867002574e-07 | 0.257091421654 | 0 | 0.0686575061037 |
| 39 | 117105 | color_cast_s2 | 2 | covariance | 0.065504305062 | 0.000297719811572 | 0.598355960209 | 0.385438769661 | -1.86963815419e-07 | 0.257091421654 | 0.0019353545073 | 0.0019353545073 |
| 40 | 563301 | clean_s0 | 2 | linear | 1.01918477079 | -0.163056617549 | -0.950698218141 | -0.956963688562 | 0.000123568203711 | 0.213120468043 | 0.80476779371 | 0.865916660135 |
| 40 | 563301 | clean_s0 | 2 | common | 0.0454203338988 | 0 | 0 | -0.902004109623 | 9.23488984208e-06 | 0.213120468043 | 0 | 0.0686575061037 |
| 40 | 563301 | clean_s0 | 2 | covariance | 0.0452933935485 | 6.35249427571e-05 | 0.900621464691 | 0.904658334479 | -4.46474433909e-08 | 0.213120468043 | 0.000330961005809 | 0.000330961005809 |
| 41 | 563301 | gamma_s1 | 2 | linear | 0.55348305369 | 1.59736858406 | 0.988971353314 | 0.990718773932 | -0.00176598126027 | 1.68066171695 | 0.961039236971 | 1.02228446452 |
| 41 | 563301 | gamma_s1 | 2 | common | 2.82462380681 | 0 | 0 | 0.920805131326 | -0.000110235047079 | 1.68066171695 | 0 | 0.0686575061037 |
| 41 | 563301 | gamma_s1 | 2 | covariance | 2.82356768696 | 0.000528124467886 | 0.874609862676 | 0.875608916135 | -5.48549372868e-07 | 1.68066171695 | 0.000359287088686 | 0.000359287088686 |
| 42 | 348315 | clean_s0 | 2 | linear | 0.158361666949 | 0.0348742885892 | 0.932651624996 | 0.223193219181 | -2.86712610029e-06 | 0.0793960243467 | 0.47096339077 | 0.403836674854 |
| 42 | 348315 | clean_s0 | 2 | common | 0.00630372868206 | 0 | 0 | -0.132331141565 | 2.89008172584e-07 | 0.0793960243467 | 0 | 0.0686575061037 |
| 42 | 348315 | clean_s0 | 2 | covariance | 0.00626313189211 | 2.03595136467e-05 | 0.733443548102 | 0.0104203867429 | -1.15889964252e-10 | 0.0793960243467 | 0.000349624574193 | 0.000349624574193 |
| 43 | 348315 | gamma_s2 | 2 | linear | 0.068701787506 | 0.0291729107336 | 0.685853781172 | 0.208834815178 | -4.93721989509e-06 | 0.127839565349 | 0.332723089828 | 0.265832329166 |
| 43 | 348315 | gamma_s2 | 2 | common | 0.0163429544687 | 0 | 0 | -0.304591844021 | 1.85985062903e-06 | 0.127839565349 | 0 | 0.0686575061037 |
| 43 | 348315 | gamma_s2 | 2 | covariance | 0.0164983838228 | -7.69348506006e-05 | -0.481885478178 | -0.0383097824866 | 4.25495967254e-09 | 0.127839565349 | 0.00124886067523 | 0.00124886067523 |
| 44 | 80341 | clean_s0 | 2 | linear | 0.176101065615 | -0.0287099982708 | -0.583881595183 | -0.260590713219 | 1.28704727495e-05 | 0.161620777829 | 0.304236410128 | 0.36312644774 |
| 44 | 80341 | clean_s0 | 2 | common | 0.0261212758259 | 0 | 0 | -0.155710020535 | 1.45405980687e-06 | 0.161620777829 | 0 | 0.0686575061037 |
| 44 | 80341 | clean_s0 | 2 | covariance | 0.0260881287527 | 1.65869714281e-05 | 0.626093055845 | 0.344131538712 | -7.67243428027e-09 | 0.161620777829 | 0.000163919649614 | 0.000163919649614 |
| 45 | 80341 | contrast_s2 | 2 | linear | 0.138272250517 | 0.0289391736383 | 0.321194772091 | 0.234015316857 | -1.71704516943e-05 | 0.369889591898 | 0.243582198856 | 0.202143791781 |
| 45 | 80341 | contrast_s2 | 2 | common | 0.136818310194 | 0 | 0 | -0.00709072741092 | 1.76707849871e-07 | 0.369889591898 | 0 | 0.0686575061037 |
| 45 | 80341 | contrast_s2 | 2 | covariance | 0.13663394252 | 9.24889903816e-05 | 0.320069327372 | 0.167206096938 | -4.74136083477e-08 | 0.369889591898 | 0.000781220930423 | 0.000781220930423 |
| 46 | 54892 | clean_s0 | 2 | linear | 0.170650532211 | 0.0215516456036 | 0.230468491687 | 0.430173004667 | -4.90690290793e-05 | 0.234790057947 | 0.398280619799 | 0.453276604154 |
| 46 | 54892 | clean_s0 | 2 | common | 0.0551263713108 | 0 | 0 | 0.373785284716 | -6.45819616517e-06 | 0.234790057947 | 0 | 0.0686575061037 |
| 46 | 54892 | clean_s0 | 2 | covariance | 0.0552918754274 | -8.21791584028e-05 | -0.326984836979 | -0.535335704688 | 1.44205422297e-07 | 0.234790057947 | 0.00107042038805 | 0.00107042038805 |
| 47 | 54892 | color_cast_s2 | 2 | linear | 0.0307600031005 | -0.00392908597538 | -0.350628029918 | 0.29722177077 | -6.17564401592e-06 | 0.0953654845129 | 0.117504278702 | 0.165207915819 |
| 47 | 54892 | color_cast_s2 | 2 | common | 0.00909457563638 | 0 | 0 | 0.662247032039 | -5.71845659548e-06 | 0.0953654845129 | 0 | 0.0686575061037 |
| 47 | 54892 | color_cast_s2 | 2 | covariance | 0.00910913842455 | -5.65677741563e-06 | -0.0329069340387 | -0.481491100945 | 1.09156474147e-07 | 0.0953654845129 | 0.00180256299059 | 0.00180256299059 |
| 48 | 449936 | clean_s0 | 3 | linear | 0.173828619944 | -0.027588550786 | -0.484439585601 | -0.408604292603 | 2.35590183144e-05 | 0.275587247059 | 0.206647496066 | 0.192604262359 |
| 48 | 449936 | clean_s0 | 3 | common | 0.0759483307415 | 0 | 0 | 0.445921718651 | -9.21626478041e-06 | 0.275587247059 | 0 | 0.069041138938 |
| 48 | 449936 | clean_s0 | 3 | covariance | 0.0759355316766 | 6.40076617744e-06 | 0.467571190994 | 0.578280270203 | -8.59907719663e-09 | 0.275587247059 | 4.9673538877e-05 | 4.9673538877e-05 |
| 49 | 449936 | gamma_s1 | 3 | linear | 0.372698574394 | -0.0197184230794 | -0.287091604188 | -0.46629036637 | 3.81921294159e-05 | 0.564311955289 | 0.121711730548 | 0.157159400946 |
| 49 | 449936 | gamma_s1 | 3 | common | 0.318447982882 | 0 | 0 | -0.584543570529 | 2.10330328799e-05 | 0.564311955289 | 0 | 0.069041138938 |
| 49 | 449936 | gamma_s1 | 3 | covariance | 0.319123147644 | -0.000337155839452 | -0.646868105787 | -0.625692607925 | 3.01185310176e-07 | 0.564311955289 | 0.000923625035512 | 0.000923625035512 |
| 50 | 534000 | clean_s0 | 3 | linear | 0.281880750365 | 0.0329945088162 | 0.207052292322 | 0.389262375924 | -7.70751980353e-05 | 0.493606297813 | 0.322835237787 | 0.356049547778 |
| 50 | 534000 | clean_s0 | 3 | common | 0.243647177241 | 0 | 0 | 0.916527785558 | -3.51896837941e-05 | 0.493606297813 | 0 | 0.069041138938 |
| 50 | 534000 | clean_s0 | 3 | covariance | 0.243502577962 | 7.23161252566e-05 | 0.806826456962 | 0.79008372863 | -7.978279097e-08 | 0.493606297813 | 0.00018158263973 | 0.00018158263973 |
| 51 | 534000 | gamma_s2 | 3 | linear | 0.364599947007 | -0.0189753314117 | -0.133452296674 | 0.0252939764894 | -4.00471396158e-06 | 0.288038565843 | 0.493642652909 | 0.533384204197 |
| 51 | 534000 | gamma_s2 | 3 | common | 0.0829662154128 | 0 | 0 | 0.241808158225 | -4.95556454842e-06 | 0.288038565843 | 0 | 0.069041138938 |
| 51 | 534000 | gamma_s2 | 3 | covariance | 0.0827842484182 | 9.19314327775e-05 | 0.231797636819 | 0.353772155058 | -1.44591336864e-07 | 0.288038565843 | 0.00137690629315 | 0.00137690629315 |
| 52 | 121632 | clean_s0 | 3 | linear | 0.0280393227861 | -0.00259135201403 | -0.230371968222 | -0.614516589834 | 4.03231676229e-06 | 0.097001748699 | 0.115962405578 | 0.132350427187 |
| 52 | 121632 | clean_s0 | 3 | common | 0.00940933925066 | 0 | 0 | -0.319109428747 | 1.09230334987e-06 | 0.097001748699 | 0 | 0.069041138938 |
| 52 | 121632 | clean_s0 | 3 | covariance | 0.0093011550582 | 5.47726767804e-05 | 0.4839824769 | 0.101853830546 | -5.89152619448e-09 | 0.097001748699 | 0.00116668808966 | 0.00116668808966 |
| 53 | 121632 | contrast_s2 | 3 | linear | 0.0412354214618 | 0.0598042017241 | 0.845239230795 | 0.797499228971 | -5.08329056833e-05 | 0.205406754887 | 0.344458836375 | 0.322080933945 |
| 53 | 121632 | contrast_s2 | 3 | common | 0.0421919349533 | 0 | 0 | 0.0636683948783 | -8.69924365089e-07 | 0.205406754887 | 0 | 0.069041138938 |
| 53 | 121632 | contrast_s2 | 3 | covariance | 0.0419180650461 | 0.000137292070274 | 0.7908801544 | 0.632189667751 | -1.05734592245e-07 | 0.205406754887 | 0.000845123259619 | 0.000845123259619 |
| 54 | 83257 | clean_s0 | 3 | linear | 0.0628183348613 | -0.0102294029079 | -0.595047628256 | -0.380372422944 | 4.65678700998e-06 | 0.0938520429176 | 0.183170202505 | 0.224808937824 |
| 54 | 83257 | clean_s0 | 3 | common | 0.00880820595981 | 0 | 0 | -0.143067115699 | 5.37912389025e-07 | 0.0938520429176 | 0 | 0.069041138938 |
| 54 | 83257 | clean_s0 | 3 | covariance | 0.00891749674776 | -5.40155532432e-05 | -0.512796118246 | -0.192224421316 | 1.17490488467e-08 | 0.0938520429176 | 0.00112235531897 | 0.00112235531897 |
| 55 | 83257 | color_cast_s2 | 3 | linear | 0.00918323094684 | 0.00229449154079 | 0.338797366836 | 0.276198872253 | -1.59586595988e-06 | 0.0901776225552 | 0.0751013343338 | 0.0630385810238 |
| 55 | 83257 | color_cast_s2 | 3 | common | 0.0081320036097 | 0 | 0 | 0.397887531394 | -2.51788771263e-06 | 0.0901776225552 | 0 | 0.069041138938 |
| 55 | 83257 | color_cast_s2 | 3 | covariance | 0.00814120907181 | -4.35129453617e-06 | -0.0680441082088 | 0.259330186345 | -1.68558371619e-08 | 0.0901776225552 | 0.000709135412391 | 0.000709135412391 |
| 56 | 80016 | clean_s0 | 3 | linear | 0.0450450280081 | 0.0350923204617 | 0.648456674182 | 0.620074275439 | -3.35453508844e-05 | 0.194537390549 | 0.278181366396 | 0.266074792606 |
| 56 | 80016 | clean_s0 | 3 | common | 0.0378447963215 | 0 | 0 | 0.294274573764 | -4.13090853043e-06 | 0.194537390549 | 0 | 0.069041138938 |
| 56 | 80016 | clean_s0 | 3 | covariance | 0.0378294677142 | 9.42036648079e-06 | 0.0258392371636 | -0.162710759934 | 6.19992148803e-08 | 0.194537390549 | 0.00187406661567 | 0.00187406661567 |
| 57 | 80016 | gamma_s1 | 3 | linear | 0.0722665730935 | 0.028037992893 | 0.436948141084 | 0.381653729957 | -2.2216625325e-05 | 0.251994590332 | 0.254639520348 | 0.233094083871 |
| 57 | 80016 | gamma_s1 | 3 | common | 0.0635012735566 | 0 | 0 | 0.105335717896 | -1.81618855841e-06 | 0.251994590332 | 0 | 0.069041138938 |
| 57 | 80016 | gamma_s1 | 3 | covariance | 0.0634568171035 | 2.31038801181e-05 | 0.0692807446302 | -0.0854901799731 | 2.82536602226e-08 | 0.251994590332 | 0.00132336961194 | 0.00132336961194 |
| 58 | 371603 | clean_s0 | 3 | linear | 0.0237123799287 | 0.107419503513 | 0.914137933945 | 0.904069678862 | -9.63700958769e-05 | 0.373802458917 | 0.314361429984 | 0.29918446727 |
| 58 | 371603 | clean_s0 | 3 | common | 0.139728278292 | 0 | 0 | -0.163025405729 | 4.01018659977e-06 | 0.373802458917 | 0 | 0.069041138938 |
| 58 | 371603 | clean_s0 | 3 | covariance | 0.139702825335 | 1.27313481428e-05 | 0.345122699868 | 0.30037721887 | -1.05615352201e-08 | 0.373802458917 | 9.86867200724e-05 | 9.86867200724e-05 |
| 59 | 371603 | gamma_s2 | 3 | linear | 2.42501602704 | 0.183629838999 | 0.50406738982 | 0.65751591968 | -0.000274336414801 | 1.65647510776 | 0.219922537266 | 0.244419752193 |
| 59 | 371603 | gamma_s2 | 3 | common | 2.74390978264 | 0 | 0 | 0.741607936991 | -8.74023688031e-05 | 1.65647510776 | 0 | 0.069041138938 |
| 59 | 371603 | gamma_s2 | 3 | covariance | 2.74044750389 | 0.00173175854095 | 0.939469871081 | 0.934287138289 | -1.77476487384e-06 | 1.65647510776 | 0.00111280627884 | 0.00111280627884 |
| 60 | 245582 | clean_s0 | 3 | linear | 0.0472989937738 | -0.00255236450389 | -0.13881335935 | 0.136581065272 | -1.77596310102e-06 | 0.103695441072 | 0.177317568974 | 0.217426765091 |
| 60 | 245582 | clean_s0 | 3 | common | 0.0107527444991 | 0 | 0 | -0.291787343855 | 1.20477191766e-06 | 0.103695441072 | 0 | 0.069041138938 |
| 60 | 245582 | clean_s0 | 3 | covariance | 0.0106790548294 | 3.73488712957e-05 | 0.358733422828 | -0.00389392554972 | 2.33810526058e-10 | 0.103695441072 | 0.00100402835307 | 0.00100402835307 |
| 61 | 245582 | contrast_s2 | 3 | linear | 0.476087662758 | -0.00546658607633 | -0.0299926526381 | -0.21314459283 | 3.45833307678e-05 | 0.614042757341 | 0.296826519641 | 0.284077257142 |
| 61 | 245582 | contrast_s2 | 3 | common | 0.377048507843 | 0 | 0 | -0.584032085667 | 2.303035789e-05 | 0.614042757341 | 0 | 0.069041138938 |
| 61 | 245582 | contrast_s2 | 3 | covariance | 0.376830192884 | 0.000109495408284 | 0.216905137239 | 0.172109890797 | -8.08145051106e-08 | 0.614042757341 | 0.000822105315092 | 0.000822105315092 |
| 62 | 212346 | clean_s0 | 3 | linear | 0.0976831483159 | -0.0166726141435 | -0.798283264122 | -0.685414470159 | 7.57778920097e-06 | 0.0877607907684 | 0.237983116276 | 0.284638204265 |
| 62 | 212346 | clean_s0 | 3 | common | 0.0077019563963 | 0 | 0 | -0.265994566839 | 7.13307471811e-07 | 0.0877607907684 | 0 | 0.069041138938 |
| 62 | 212346 | clean_s0 | 3 | covariance | 0.00769957446521 | 1.21321385737e-06 | 0.0655349988668 | 0.120650812721 | -9.88529991428e-10 | 0.0877607907684 | 0.000210942232551 | 0.000210942232551 |
| 63 | 212346 | color_cast_s2 | 3 | linear | 0.0945316566777 | 0.00125571473145 | 0.0454215524719 | 0.073734192106 | -8.4899026039e-07 | 0.0929844450384 | 0.297316294746 | 0.307947420605 |
| 63 | 212346 | color_cast_s2 | 3 | common | 0.00864610701909 | 0 | 0 | -0.480620693931 | 1.24070240605e-06 | 0.0929844450384 | 0 | 0.069041138938 |
| 63 | 212346 | color_cast_s2 | 3 | covariance | 0.00852558676004 | 6.05579622972e-05 | 0.843839227453 | 0.925264471541 | -2.67008038829e-08 | 0.0929844450384 | 0.000771793718031 | 0.000771793718031 |

## Predeclared gate

| Flag | Passed |
| --- | --- |
| pooled_R2_at_least_point10 | False |
| at_least_three_positive_R2_folds | False |
| median_residual_cosine_at_least_point20 | False |
| positive_dot_at_least44 | False |
| clean_positive_dot_at_least20 | False |
| corrupted_positive_dot_at_least20 | False |
| R2_above_null95 | True |
| cosine_above_null95 | False |

Decision: current_linear_initialization_branch_not_supported_request_research_pivot

Full input/residual/prediction/delta vectors, SVD coefficients and fold indices are in observed.json. Each permutation_NNN.json retains all four fitted coefficients, pair/row mappings, held-out residual/full-gradient predictions and pooled metrics. No null outcome is discarded.
