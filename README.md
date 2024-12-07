# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py project/parallel_check.py tests/test_tensor_general.py


Sentiment Training:

Upped Training Size to 2500
Found that the bigger set helped to raise maximum accuracy much higher
Upped training rate to 0.014


Epoch 1, loss 173.32844876411022, train accuracy: 45.84%
Validation accuracy: 52.00%
Best Valid accuracy: 52.00%
Epoch 2, loss 171.81089620670113, train accuracy: 53.52%
Validation accuracy: 53.00%
Best Valid accuracy: 53.00%
Epoch 3, loss 167.41528895501804, train accuracy: 56.64%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 4, loss 161.23366794759082, train accuracy: 60.40%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 5, loss 156.79649044510458, train accuracy: 63.04%
Validation accuracy: 67.00%
Best Valid accuracy: 74.00%
Epoch 6, loss 150.98378817000054, train accuracy: 66.40%
Validation accuracy: 66.00%
Best Valid accuracy: 74.00%
Epoch 7, loss 148.3166177779836, train accuracy: 67.84%
Validation accuracy: 65.00%
Best Valid accuracy: 74.00%
Epoch 8, loss 144.80555059030988, train accuracy: 69.68%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 9, loss 143.55616802191838, train accuracy: 69.48%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 10, loss 139.44593378745725, train accuracy: 71.88%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 11, loss 138.28556597059492, train accuracy: 71.68%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 12, loss 136.5527926008373, train accuracy: 72.08%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 13, loss 134.38544455117847, train accuracy: 72.92%
Validation accuracy: 82.00%
Best Valid accuracy: 82.00%
Epoch 14, loss 132.0806051456242, train accuracy: 73.60%
Validation accuracy: 75.00%
Best Valid accuracy: 82.00%
Epoch 15, loss 129.49165661896484, train accuracy: 74.68%
Validation accuracy: 78.00%
Best Valid accuracy: 82.00%
Epoch 16, loss 127.49031530672742, train accuracy: 75.36%
Validation accuracy: 74.00%
Best Valid accuracy: 82.00%
Epoch 17, loss 127.68665227110289, train accuracy: 74.68%
Validation accuracy: 76.00%
Best Valid accuracy: 82.00%
Epoch 18, loss 128.04867424135102, train accuracy: 74.52%
Validation accuracy: 78.00%
Best Valid accuracy: 82.00%
Epoch 19, loss 125.48817726962756, train accuracy: 75.88%
Validation accuracy: 77.00%
Best Valid accuracy: 82.00%
Epoch 20, loss 123.5447739713782, train accuracy: 76.40%
Validation accuracy: 78.00%
Best Valid accuracy: 82.00%
Epoch 21, loss 121.71763019718593, train accuracy: 77.52%
Validation accuracy: 84.00%
Best Valid accuracy: 84.00%
Epoch 22, loss 119.8586044124365, train accuracy: 78.60%
Validation accuracy: 77.00%
Best Valid accuracy: 84.00%
Epoch 23, loss 119.8384541706896, train accuracy: 78.12%
Validation accuracy: 78.00%
Best Valid accuracy: 84.00%
Epoch 24, loss 120.94913639368487, train accuracy: 77.24%
Validation accuracy: 80.00%
Best Valid accuracy: 84.00%
Epoch 25, loss 120.70141022451992, train accuracy: 77.32%
Validation accuracy: 79.00%
Best Valid accuracy: 84.00%
Epoch 26, loss 119.48130604509066, train accuracy: 78.24%
Validation accuracy: 84.00%
Best Valid accuracy: 84.00%
Epoch 27, loss 118.57622913214928, train accuracy: 78.56%
Validation accuracy: 78.00%
Best Valid accuracy: 84.00%
Epoch 28, loss 116.46148309863119, train accuracy: 79.44%
Validation accuracy: 79.00%
Best Valid accuracy: 84.00%
Epoch 29, loss 115.36249081761754, train accuracy: 79.88%
Validation accuracy: 80.00%
Best Valid accuracy: 84.00%
Epoch 30, loss 113.96221452893607, train accuracy: 80.84%
Validation accuracy: 83.00%
Best Valid accuracy: 84.00%
Epoch 31, loss 115.61373573772457, train accuracy: 79.52%
Validation accuracy: 78.00%
Best Valid accuracy: 84.00%
Epoch 32, loss 113.6535472076802, train accuracy: 80.88%
Validation accuracy: 78.00%
Best Valid accuracy: 84.00%
Epoch 33, loss 113.47015465056158, train accuracy: 80.64%
Validation accuracy: 80.00%
Best Valid accuracy: 84.00%
Epoch 34, loss 112.48308291566595, train accuracy: 81.56%
Validation accuracy: 79.00%
Best Valid accuracy: 84.00%
Epoch 35, loss 113.52558309597696, train accuracy: 80.68%
Validation accuracy: 82.00%
Best Valid accuracy: 84.00%


MNIST Data Log:

Epoch 1 loss 2.2984768342832163 valid acc 2/16
Epoch 1 loss 11.553925689604057 valid acc 2/16
Epoch 1 loss 11.515060041916593 valid acc 4/16
Epoch 1 loss 11.505379168868187 valid acc 3/16
Epoch 1 loss 11.542747566852926 valid acc 3/16
Epoch 1 loss 11.532982708273169 valid acc 5/16
Epoch 1 loss 11.476202555167827 valid acc 3/16
Epoch 1 loss 11.452640027592345 valid acc 3/16
Epoch 1 loss 11.427070491742112 valid acc 5/16
Epoch 1 loss 11.388702396623403 valid acc 5/16
Epoch 1 loss 11.207149129423904 valid acc 7/16
Epoch 1 loss 10.927587651230624 valid acc 10/16
Epoch 1 loss 10.415433053724755 valid acc 5/16
Epoch 1 loss 9.834804634031046 valid acc 10/16
Epoch 1 loss 9.254480046669723 valid acc 10/16
Epoch 1 loss 7.616678292400064 valid acc 11/16
Epoch 1 loss 8.895546436982649 valid acc 9/16
Epoch 1 loss 7.283428701160489 valid acc 11/16
Epoch 1 loss 7.180714080254626 valid acc 14/16
Epoch 1 loss 5.937638642554352 valid acc 13/16
Epoch 1 loss 5.835557268706056 valid acc 13/16
Epoch 1 loss 4.992024282005094 valid acc 11/16
Epoch 1 loss 3.932917488700406 valid acc 10/16
Epoch 1 loss 5.033370079347107 valid acc 14/16
Epoch 1 loss 4.660189849292876 valid acc 11/16
Epoch 1 loss 5.364029261283151 valid acc 11/16
Epoch 1 loss 4.417164486428854 valid acc 13/16
Epoch 1 loss 3.733989085436286 valid acc 13/16
Epoch 1 loss 3.965385555611059 valid acc 10/16
Epoch 1 loss 3.192996348326669 valid acc 13/16
Epoch 1 loss 4.649408511487021 valid acc 13/16
Epoch 1 loss 4.158891402306859 valid acc 11/16
Epoch 1 loss 3.017021189003046 valid acc 10/16
Epoch 1 loss 4.5298381361555435 valid acc 13/16
Epoch 1 loss 6.048364836684 valid acc 13/16
Epoch 1 loss 4.556658408053839 valid acc 12/16
Epoch 1 loss 3.16069661395811 valid acc 11/16
Epoch 1 loss 3.164427940432012 valid acc 13/16
Epoch 1 loss 3.53136602761624 valid acc 13/16
Epoch 1 loss 3.766972965046878 valid acc 13/16
Epoch 1 loss 3.17378832262645 valid acc 14/16
Epoch 1 loss 3.182676094422497 valid acc 15/16
Epoch 1 loss 3.4969660970752607 valid acc 14/16
Epoch 1 loss 3.1491121344009647 valid acc 13/16
Epoch 1 loss 3.978612861494021 valid acc 16/16
Epoch 1 loss 2.5786774534692762 valid acc 14/16
Epoch 1 loss 3.7628748080354226 valid acc 12/16
Epoch 1 loss 3.4672872600595404 valid acc 15/16
Epoch 1 loss 3.1198676696632726 valid acc 14/16
Epoch 1 loss 2.9247059133164814 valid acc 15/16
Epoch 1 loss 4.92099403891371 valid acc 15/16
Epoch 1 loss 3.612996351561772 valid acc 16/16
Epoch 1 loss 3.676365974797543 valid acc 15/16
Epoch 1 loss 2.6365169186826853 valid acc 15/16
Epoch 1 loss 3.350175564735807 valid acc 14/16
Epoch 1 loss 2.2139417307309865 valid acc 14/16
Epoch 1 loss 3.0020045387161094 valid acc 12/16
Epoch 1 loss 3.368523781681946 valid acc 13/16
Epoch 1 loss 2.776792475562422 valid acc 14/16
Epoch 1 loss 2.8888368385731598 valid acc 13/16
Epoch 1 loss 3.3920711169366875 valid acc 13/16
Epoch 1 loss 3.1947345793033177 valid acc 14/16
Epoch 1 loss 3.455795238935472 valid acc 13/16
Epoch 2 loss 0.46514464142241396 valid acc 13/16
Epoch 2 loss 2.4975095560535916 valid acc 13/16
Epoch 2 loss 3.1737950957614784 valid acc 14/16
Epoch 2 loss 2.547948775082217 valid acc 14/16
Epoch 2 loss 2.880743313461971 valid acc 14/16
Epoch 2 loss 2.7987415152084703 valid acc 15/16
Epoch 2 loss 2.7744206145960963 valid acc 13/16
Epoch 2 loss 3.0366028922872066 valid acc 13/16
Epoch 2 loss 3.222251134715582 valid acc 14/16
Epoch 2 loss 2.461122668878308 valid acc 15/16
Epoch 2 loss 2.5220735502381615 valid acc 15/16
Epoch 2 loss 3.07362457636125 valid acc 14/16
Epoch 2 loss 3.0821319350583574 valid acc 14/16
Epoch 2 loss 3.807784033548715 valid acc 14/16
Epoch 2 loss 4.133165142868199 valid acc 13/16
Epoch 2 loss 2.65974660599764 valid acc 14/16
Epoch 2 loss 3.5552781720619606 valid acc 14/16
Epoch 2 loss 3.5885028192048303 valid acc 15/16
Epoch 2 loss 2.384960383213064 valid acc 14/16
Epoch 2 loss 2.140822767092392 valid acc 14/16
Epoch 2 loss 3.1382171168209516 valid acc 14/16
Epoch 2 loss 2.78607006174703 valid acc 12/16
Epoch 2 loss 2.0838403984361014 valid acc 15/16
Epoch 2 loss 2.480428027895944 valid acc 13/16
Epoch 2 loss 2.5347328598651506 valid acc 14/16
Epoch 2 loss 1.7918350851579992 valid acc 14/16
Epoch 2 loss 1.989646487733538 valid acc 14/16
Epoch 2 loss 2.2793580914958937 valid acc 15/16
Epoch 2 loss 1.9547280321511311 valid acc 14/16
Epoch 2 loss 1.1255990439195673 valid acc 14/16
Epoch 2 loss 2.5840819585093224 valid acc 14/16
Epoch 2 loss 2.400375594302148 valid acc 14/16
Epoch 2 loss 1.3061174164485232 valid acc 12/16
Epoch 2 loss 1.6630567856535372 valid acc 16/16
Epoch 2 loss 3.8675611199859765 valid acc 16/16
Epoch 2 loss 2.70834551269025 valid acc 15/16
Epoch 2 loss 2.0338990200924956 valid acc 13/16
Epoch 2 loss 2.6471945146819515 valid acc 14/16
Epoch 2 loss 2.3833366306300237 valid acc 14/16
Epoch 2 loss 2.0554089522768972 valid acc 14/16
Epoch 2 loss 1.4433228430330491 valid acc 14/16
Epoch 2 loss 2.0328620378980866 valid acc 15/16
Epoch 2 loss 2.3046397667621488 valid acc 14/16
Epoch 2 loss 1.9748247787219184 valid acc 15/16
Epoch 2 loss 3.125403952569803 valid acc 14/16
Epoch 2 loss 1.610702169069423 valid acc 15/16
Epoch 2 loss 1.5178888325730069 valid acc 14/16
Epoch 2 loss 2.8176505946357415 valid acc 14/16
Epoch 2 loss 2.014517932158569 valid acc 15/16
Epoch 2 loss 1.3552590974018934 valid acc 14/16
Epoch 2 loss 1.9749550905620508 valid acc 14/16
Epoch 2 loss 1.8827894985208493 valid acc 15/16
Epoch 2 loss 2.802179753174381 valid acc 14/16
Epoch 2 loss 2.071352460972962 valid acc 14/16
Epoch 2 loss 2.633516891016197 valid acc 14/16
Epoch 2 loss 1.550855646890585 valid acc 15/16
Epoch 2 loss 1.9893657938730862 valid acc 13/16
Epoch 2 loss 1.9441741165617104 valid acc 13/16
Epoch 2 loss 2.676451855526084 valid acc 15/16
Epoch 2 loss 2.15404640679494 valid acc 14/16
Epoch 2 loss 1.9397380377549875 valid acc 13/16
Epoch 2 loss 1.6203442885446182 valid acc 15/16
Epoch 2 loss 2.51468740169127 valid acc 15/16
Epoch 3 loss 0.12968347884326842 valid acc 15/16
Epoch 3 loss 1.9143656165171767 valid acc 15/16
Epoch 3 loss 1.9391568122400886 valid acc 15/16
Epoch 3 loss 2.2718397489774893 valid acc 14/16
Epoch 3 loss 1.2417719845814752 valid acc 14/16
Epoch 3 loss 1.5589481865720853 valid acc 15/16
Epoch 3 loss 2.155579376470664 valid acc 14/16
Epoch 3 loss 2.0618327397464573 valid acc 15/16
Epoch 3 loss 2.3866946090260175 valid acc 15/16
Epoch 3 loss 1.321592820874362 valid acc 15/16
Epoch 3 loss 1.7386251925657894 valid acc 15/16
Epoch 3 loss 2.761977748532402 valid acc 14/16
Epoch 3 loss 2.5373175201544447 valid acc 14/16
Epoch 3 loss 2.5961949680215173 valid acc 14/16
Epoch 3 loss 3.6786520391842066 valid acc 13/16
Epoch 3 loss 1.9084428648083582 valid acc 13/16
Epoch 3 loss 3.292306191036185 valid acc 15/16
Epoch 3 loss 2.1661429941553183 valid acc 15/16
Epoch 3 loss 1.5701859255023778 valid acc 15/16
Epoch 3 loss 1.4816547697876201 valid acc 15/16
Epoch 3 loss 2.179402945876836 valid acc 13/16
Epoch 3 loss 2.016529526106551 valid acc 13/16
Epoch 3 loss 1.2971093360462953 valid acc 15/16
Epoch 3 loss 1.0838563923413194 valid acc 14/16
Epoch 3 loss 0.9789607911560911 valid acc 14/16
Epoch 3 loss 1.4305940765753844 valid acc 15/16
Epoch 3 loss 1.608826370273884 valid acc 15/16
Epoch 3 loss 1.7050284945241985 valid acc 14/16
Epoch 3 loss 1.358281881239677 valid acc 15/16
Epoch 3 loss 1.0274066344846016 valid acc 14/16
Epoch 3 loss 1.5935174927156517 valid acc 14/16
Epoch 3 loss 1.5520477909375483 valid acc 14/16
Epoch 3 loss 1.2511426688910596 valid acc 14/16
Epoch 3 loss 1.133971893784828 valid acc 14/16
Epoch 3 loss 2.281926470236887 valid acc 16/16
Epoch 3 loss 2.6679502417119743 valid acc 13/16
Epoch 3 loss 1.5114770615914765 valid acc 14/16
Epoch 3 loss 1.4544075047184966 valid acc 14/16
Epoch 3 loss 1.8419360696698885 valid acc 15/16
Epoch 3 loss 1.892435327508327 valid acc 14/16
Epoch 3 loss 1.2443400202224693 valid acc 15/16
Epoch 3 loss 1.214590666681804 valid acc 15/16
Epoch 3 loss 1.6621177629548496 valid acc 14/16
Epoch 3 loss 1.1893788938580128 valid acc 15/16
Epoch 3 loss 2.240395874606663 valid acc 13/16
Epoch 3 loss 1.3473069587294306 valid acc 16/16
Epoch 3 loss 2.1173459545452973 valid acc 14/16
Epoch 3 loss 2.6813703739454247 valid acc 15/16
Epoch 3 loss 1.6243688776624576 valid acc 15/16
Epoch 3 loss 1.6962181093928248 valid acc 15/16
Epoch 3 loss 1.3767239309563815 valid acc 15/16
Epoch 3 loss 1.9136971605627062 valid acc 16/16
Epoch 3 loss 2.196450217592849 valid acc 14/16
Epoch 3 loss 1.5059877806192377 valid acc 14/16
Epoch 3 loss 2.297931090583152 valid acc 14/16
Epoch 3 loss 1.2849553647408996 valid acc 15/16
Epoch 3 loss 1.083463584838929 valid acc 15/16
Epoch 3 loss 1.8855586181790995 valid acc 15/16
Epoch 3 loss 1.4770853136490791 valid acc 13/16
Epoch 3 loss 1.6179369077032826 valid acc 15/16
Epoch 3 loss 1.3402929613917927 valid acc 13/16
Epoch 3 loss 1.3012014533546563 valid acc 14/16
Epoch 3 loss 2.5709503238900515 valid acc 14/16
Epoch 4 loss 0.015013622217926903 valid acc 14/16
Epoch 4 loss 1.3850892291695853 valid acc 15/16
Epoch 4 loss 2.8085313729630297 valid acc 14/16
Epoch 4 loss 2.44714980744655 valid acc 14/16
Epoch 4 loss 1.669462275168506 valid acc 14/16
Epoch 4 loss 1.2373565195150606 valid acc 14/16
Epoch 4 loss 1.8024415164290937 valid acc 13/16
Epoch 4 loss 1.6148624568383678 valid acc 15/16
Epoch 4 loss 1.31023035370203 valid acc 15/16
Epoch 4 loss 1.1365631406679844 valid acc 15/16
Epoch 4 loss 1.3101540814747383 valid acc 15/16
Epoch 4 loss 1.9530036370826065 valid acc 15/16
Epoch 4 loss 1.8748790184502648 valid acc 15/16
Epoch 4 loss 2.542155002210328 valid acc 14/16
Epoch 4 loss 2.1295921452273356 valid acc 15/16
Epoch 4 loss 1.4863262586353048 valid acc 14/16
Epoch 4 loss 2.5584717603321274 valid acc 15/16
Epoch 4 loss 1.91066892766638 valid acc 15/16
Epoch 4 loss 1.4664797541719157 valid acc 14/16
Epoch 4 loss 1.566651165196994 valid acc 14/16
Epoch 4 loss 1.7234979682696645 valid acc 12/16
Epoch 4 loss 1.061186544223969 valid acc 13/16
Epoch 4 loss 0.6070095225361649 valid acc 15/16
Epoch 4 loss 1.3583287035439324 valid acc 14/16
Epoch 4 loss 0.9236451813041944 valid acc 15/16
Epoch 4 loss 0.9471306993877207 valid acc 15/16
Epoch 4 loss 1.3423100009437707 valid acc 16/16
Epoch 4 loss 1.0903097310137655 valid acc 16/16
Epoch 4 loss 1.2915359250624634 valid acc 15/16
Epoch 4 loss 0.6017373626510092 valid acc 15/16
Epoch 4 loss 1.2703998034300408 valid acc 14/16
Epoch 4 loss 0.9324443198867818 valid acc 14/16
Epoch 4 loss 0.9466000396518028 valid acc 13/16
Epoch 4 loss 1.2340455845655423 valid acc 14/16
Epoch 4 loss 1.8441405125789863 valid acc 15/16
Epoch 4 loss 1.3051397195507686 valid acc 15/16
Epoch 4 loss 1.1111373553828117 valid acc 15/16
Epoch 4 loss 1.2285449373914559 valid acc 15/16
Epoch 4 loss 1.290542472178463 valid acc 15/16
Epoch 4 loss 1.0223982239873164 valid acc 14/16
Epoch 4 loss 0.969454361504573 valid acc 14/16
Epoch 4 loss 1.158792954487024 valid acc 15/16
Epoch 4 loss 1.0990220819475183 valid acc 14/16
Epoch 4 loss 0.8801193492351613 valid acc 13/16
Epoch 4 loss 2.523575559089373 valid acc 14/16
Epoch 4 loss 1.1296962873207335 valid acc 15/16
Epoch 4 loss 2.074273994567488 valid acc 14/16
Epoch 4 loss 3.261954267485604 valid acc 15/16
Epoch 4 loss 1.1058936923382827 valid acc 16/16
Epoch 4 loss 1.3008861052822973 valid acc 16/16
Epoch 4 loss 1.2282455874523206 valid acc 14/16
Epoch 4 loss 1.502606319086835 valid acc 15/16
Epoch 4 loss 2.1022374082895756 valid acc 15/16
Epoch 4 loss 1.4368173744917299 valid acc 14/16
Epoch 4 loss 2.1887747736371432 valid acc 15/16
Epoch 4 loss 1.1246208423323312 valid acc 15/16
Epoch 4 loss 1.0581459008842158 valid acc 15/16
Epoch 4 loss 1.305637672032212 valid acc 14/16
Epoch 4 loss 1.30033876474828 valid acc 14/16
Epoch 4 loss 1.2826606017550803 valid acc 14/16
Epoch 4 loss 1.2518273987490807 valid acc 14/16
Epoch 4 loss 0.8090304695347794 valid acc 14/16
Epoch 4 loss 1.3506690529971226 valid acc 14/16
Epoch 5 loss 0.012236083011093912 valid acc 15/16
Epoch 5 loss 1.3734912637742818 valid acc 14/16
Epoch 5 loss 0.901934658898846 valid acc 14/16
Epoch 5 loss 1.0154902760634714 valid acc 14/16
Epoch 5 loss 0.7667338495834698 valid acc 14/16
Epoch 5 loss 0.7684946171152157 valid acc 13/16
Epoch 5 loss 1.4098066861657204 valid acc 14/16
Epoch 5 loss 1.4338528432512194 valid acc 13/16
Epoch 5 loss 1.6120354110349242 valid acc 14/16
Epoch 5 loss 1.089119667746102 valid acc 14/16
Epoch 5 loss 1.491520908560688 valid acc 15/16
Epoch 5 loss 2.2137600925596357 valid acc 15/16
Epoch 5 loss 1.3969416379841793 valid acc 15/16
Epoch 5 loss 1.6924681168504585 valid acc 13/16
Epoch 5 loss 1.6476206847186938 valid acc 15/16
Epoch 5 loss 1.296025319396865 valid acc 13/16
Epoch 5 loss 2.1081941131883557 valid acc 14/16
Epoch 5 loss 1.601744561931813 valid acc 15/16
Epoch 5 loss 1.3275843822190614 valid acc 15/16
Epoch 5 loss 1.45657581059341 valid acc 15/16
Epoch 5 loss 1.8003044889041484 valid acc 14/16
Epoch 5 loss 1.4681598939274676 valid acc 13/16
Epoch 5 loss 0.8994360553781667 valid acc 16/16
Epoch 5 loss 1.6498472685966137 valid acc 14/16
Epoch 5 loss 0.9094990325453325 valid acc 14/16
Epoch 5 loss 0.754887017793076 valid acc 15/16
Epoch 5 loss 0.39318321174008153 valid acc 16/16
Epoch 5 loss 1.2960722571892214 valid acc 16/16
Epoch 5 loss 0.9843327861291316 valid acc 14/16
Epoch 5 loss 0.3387379091029636 valid acc 16/16
Epoch 5 loss 1.5540930451763342 valid acc 15/16
Epoch 5 loss 1.051819909767982 valid acc 15/16
Epoch 5 loss 0.8157455101381443 valid acc 15/16
Epoch 5 loss 1.0238565722524264 valid acc 15/16
Epoch 5 loss 1.5213069516548847 valid acc 15/16
Epoch 5 loss 1.2671249652586036 valid acc 15/16
Epoch 5 loss 1.237943659288836 valid acc 13/16
Epoch 5 loss 0.7685307015274555 valid acc 16/16
Epoch 5 loss 0.8458601267657896 valid acc 16/16
Epoch 5 loss 1.0856960311550337 valid acc 14/16
Epoch 5 loss 0.9308432242891904 valid acc 16/16
Epoch 5 loss 1.3091437554172691 valid acc 16/16
Epoch 5 loss 1.4110816690118593 valid acc 15/16
Epoch 5 loss 0.9403720086179015 valid acc 15/16
Epoch 5 loss 1.4032337892439652 valid acc 16/16
Epoch 5 loss 1.1424136872379878 valid acc 15/16
Epoch 5 loss 1.13969608875655 valid acc 15/16
Epoch 5 loss 1.5078559262425495 valid acc 16/16
Epoch 5 loss 0.8101950517890371 valid acc 16/16
Epoch 5 loss 0.7286101508220211 valid acc 16/16
Epoch 5 loss 0.6252486874129317 valid acc 16/16
Epoch 5 loss 1.115399128257062 valid acc 16/16
Epoch 5 loss 1.6604814350037336 valid acc 14/16
Epoch 5 loss 1.1103890986868474 valid acc 15/16
Epoch 5 loss 1.4710796161871424 valid acc 15/16
Epoch 5 loss 0.8533363334893203 valid acc 15/16
Epoch 5 loss 0.9311277235775094 valid acc 16/16
Epoch 5 loss 0.8635186340280181 valid acc 15/16
Epoch 5 loss 1.2470494212777647 valid acc 15/16
Epoch 5 loss 0.8914392407350105 valid acc 15/16
Epoch 5 loss 0.9378232672044242 valid acc 16/16
Epoch 5 loss 0.764807390086754 valid acc 16/16
Epoch 5 loss 1.5243030926760337 valid acc 15/16
Epoch 6 loss 0.014853854562100754 valid acc 16/16
Epoch 6 loss 1.0749808121704423 valid acc 16/16
Epoch 6 loss 1.2044729219704187 valid acc 16/16
Epoch 6 loss 0.9948089988435006 valid acc 15/16
Epoch 6 loss 0.7541411606233153 valid acc 16/16
Epoch 6 loss 0.5162913985901204 valid acc 16/16
Epoch 6 loss 0.9465863698781682 valid acc 15/16
Epoch 6 loss 0.9422481587549365 valid acc 16/16
Epoch 6 loss 1.4858817683134926 valid acc 16/16
Epoch 6 loss 0.73990901848598 valid acc 16/16
Epoch 6 loss 1.0603470915942865 valid acc 16/16
Epoch 6 loss 1.1864360725140388 valid acc 16/16
Epoch 6 loss 1.5111228792745885 valid acc 15/16
Epoch 6 loss 1.4662549140275178 valid acc 16/16
Epoch 6 loss 1.6573425710055425 valid acc 15/16
Epoch 6 loss 1.2609251690320025 valid acc 15/16
Epoch 6 loss 1.4666616993436647 valid acc 14/16
Epoch 6 loss 1.579248904835221 valid acc 15/16
Epoch 6 loss 0.9565439772823681 valid acc 16/16
Epoch 6 loss 0.9994269158016672 valid acc 14/16
Epoch 6 loss 1.3602640530749461 valid acc 15/16
Epoch 6 loss 0.6649331127581921 valid acc 13/16
Epoch 6 loss 0.22476501853956626 valid acc 15/16
Epoch 6 loss 1.073539142976561 valid acc 13/16
Epoch 6 loss 1.123411822580438 valid acc 13/16
Epoch 6 loss 0.985474623216636 valid acc 15/16
Epoch 6 loss 0.650331712459202 valid acc 16/16
Epoch 6 loss 1.1242127835518378 valid acc 15/16
Epoch 6 loss 0.7597301692982183 valid acc 14/16
Epoch 6 loss 0.5809198496492461 valid acc 15/16
Epoch 6 loss 1.464557056059662 valid acc 15/16
Epoch 6 loss 1.436796809936197 valid acc 15/16
Epoch 6 loss 0.7026002356916458 valid acc 15/16
Epoch 6 loss 0.9304464271421025 valid acc 15/16
Epoch 6 loss 1.6358579242660607 valid acc 15/16
Epoch 6 loss 1.462255028066615 valid acc 14/16
Epoch 6 loss 0.9107974083087896 valid acc 13/16
Epoch 6 loss 0.888216477393964 valid acc 15/16
Epoch 6 loss 1.2581516593646054 valid acc 15/16
Epoch 6 loss 1.406297929770509 valid acc 15/16
Epoch 6 loss 0.5446967473349518 valid acc 15/16
Epoch 6 loss 1.4412691185323334 valid acc 16/16
Epoch 6 loss 1.3263773450610554 valid acc 15/16
Epoch 6 loss 1.3300113788141257 valid acc 15/16
Epoch 6 loss 1.3131177991912786 valid acc 13/16
Epoch 6 loss 0.9297076276471238 valid acc 16/16
Epoch 6 loss 0.622149346674324 valid acc 16/16
Epoch 6 loss 1.466381149554871 valid acc 15/16
Epoch 6 loss 0.5924401984698084 valid acc 15/16
Epoch 6 loss 0.6992360532286255 valid acc 16/16
Epoch 6 loss 0.2438067294013229 valid acc 16/16
Epoch 6 loss 1.0442632663411069 valid acc 16/16
Epoch 6 loss 1.2284238898283004 valid acc 15/16
Epoch 6 loss 0.618645513957212 valid acc 16/16
Epoch 6 loss 1.146352137174806 valid acc 16/16
Epoch 6 loss 0.6658173416001162 valid acc 16/16
Epoch 6 loss 0.6567674410902034 valid acc 16/16
Epoch 6 loss 0.4899052894982024 valid acc 16/16
Epoch 6 loss 1.298670640864513 valid acc 16/16
Epoch 6 loss 0.8721511850281597 valid acc 15/16
Epoch 6 loss 0.8508232335763014 valid acc 15/16
Epoch 6 loss 0.6664290621884319 valid acc 16/16
Epoch 6 loss 1.2242742128803563 valid acc 15/16
Epoch 7 loss 0.018413610543594805 valid acc 16/16
Epoch 7 loss 1.0923146127894345 valid acc 15/16
Epoch 7 loss 1.1360287129433684 valid acc 15/16
Epoch 7 loss 1.0285391267693353 valid acc 16/16
Epoch 7 loss 0.5674492447104482 valid acc 15/16
Epoch 7 loss 0.5871385105750563 valid acc 15/16
Epoch 7 loss 1.040592261399005 valid acc 14/16
Epoch 7 loss 0.9938332081771419 valid acc 14/16
Epoch 7 loss 0.6852784890480937 valid acc 14/16
Epoch 7 loss 0.7662041258574711 valid acc 15/16
Epoch 7 loss 0.9903613689298081 valid acc 15/16
Epoch 7 loss 1.5000697689436775 valid acc 15/16
Epoch 7 loss 1.029210104550675 valid acc 14/16
Epoch 7 loss 0.7576113814575369 valid acc 14/16
Epoch 7 loss 1.3032850531651057 valid acc 16/16
Epoch 7 loss 1.4259441446310284 valid acc 15/16
Epoch 7 loss 1.783137272420819 valid acc 15/16
Epoch 7 loss 1.712412725637015 valid acc 15/16
Epoch 7 loss 1.2490104183139648 valid acc 16/16
Epoch 7 loss 1.1478104703548688 valid acc 14/16
Epoch 7 loss 1.154497701249927 valid acc 13/16
Epoch 7 loss 0.7090437709769842 valid acc 13/16
Epoch 7 loss 0.48396770444950765 valid acc 13/16
Epoch 7 loss 0.9603388640720485 valid acc 13/16
Epoch 7 loss 0.444027940888058 valid acc 13/16
Epoch 7 loss 0.7829857790086252 valid acc 15/16
Epoch 7 loss 0.6472956073852636 valid acc 16/16
Epoch 7 loss 1.055202137913023 valid acc 15/16
Epoch 7 loss 0.44731254289719397 valid acc 14/16
Epoch 7 loss 0.3816839104694271 valid acc 15/16
Epoch 7 loss 1.0639486568354781 valid acc 15/16
Epoch 7 loss 0.6644668114104703 valid acc 14/16
Epoch 7 loss 0.5483464150214736 valid acc 15/16
Epoch 7 loss 1.025338957040558 valid acc 15/16
Epoch 7 loss 1.510681286625588 valid acc 15/16
Epoch 7 loss 1.1919108736927033 valid acc 15/16
Epoch 7 loss 0.6820376600062267 valid acc 15/16
Epoch 7 loss 0.5453887575973557 valid acc 15/16
Epoch 7 loss 0.7401342467026388 valid acc 15/16
Epoch 7 loss 1.0742307109423057 valid acc 15/16
Epoch 7 loss 0.3326902830149191 valid acc 16/16
Epoch 7 loss 0.7173811168312355 valid acc 15/16
Epoch 7 loss 0.5943722581684352 valid acc 16/16
Epoch 7 loss 1.3252018026018177 valid acc 15/16
Epoch 7 loss 1.5860651971542845 valid acc 14/16
Epoch 7 loss 0.5480052834535762 valid acc 15/16
Epoch 7 loss 1.4686689380494942 valid acc 16/16
Epoch 7 loss 1.8976042755648759 valid acc 16/16
Epoch 7 loss 1.0105237446717852 valid acc 16/16
Epoch 7 loss 0.8945563877699125 valid acc 16/16
Epoch 7 loss 0.6991834434925206 valid acc 16/16
Epoch 7 loss 0.6262083591065613 valid acc 16/16
Epoch 7 loss 1.0742751127475465 valid acc 16/16
Epoch 7 loss 0.871518814714211 valid acc 14/16
Epoch 7 loss 1.3387442925818283 valid acc 15/16
Epoch 7 loss 0.7785331408338185 valid acc 16/16
Epoch 7 loss 1.0419508183211064 valid acc 16/16
Epoch 7 loss 1.1080826275048095 valid acc 16/16
Epoch 7 loss 0.9701935473323964 valid acc 16/16
Epoch 7 loss 0.7445706449032029 valid acc 15/16
Epoch 7 loss 0.781325065460994 valid acc 14/16
Epoch 7 loss 0.9046337579561226 valid acc 15/16
Epoch 7 loss 1.1522205859035743 valid acc 14/16
Epoch 8 loss 0.008705472419425264 valid acc 14/16
Epoch 8 loss 1.2127748320027127 valid acc 14/16
Epoch 8 loss 1.0674982448574952 valid acc 15/16
Epoch 8 loss 0.7413592918557532 valid acc 14/16
Epoch 8 loss 1.0590758563872704 valid acc 15/16
Epoch 8 loss 0.3481676694573848 valid acc 15/16
Epoch 8 loss 0.9385094013594917 valid acc 15/16
Epoch 8 loss 0.7688882759548823 valid acc 15/16
Epoch 8 loss 0.57089844782839 valid acc 15/16
Epoch 8 loss 0.6730255307024438 valid acc 15/16
Epoch 8 loss 0.7674849060729344 valid acc 15/16
Epoch 8 loss 1.1420496880502506 valid acc 16/16
Epoch 8 loss 0.7553444602426395 valid acc 16/16
Epoch 8 loss 1.139794059354475 valid acc 15/16
Epoch 8 loss 1.5564957222975895 valid acc 16/16
Epoch 8 loss 1.006793513734066 valid acc 14/16
Epoch 8 loss 1.1744496220596754 valid acc 15/16
Epoch 8 loss 1.0232924894416755 valid acc 15/16
Epoch 8 loss 0.9694823393441994 valid acc 14/16
Epoch 8 loss 0.4492252007736006 valid acc 14/16
Epoch 8 loss 1.3843809403027194 valid acc 13/16
Epoch 8 loss 0.6558461353147237 valid acc 13/16
Epoch 8 loss 0.21453291652041517 valid acc 13/16
Epoch 8 loss 0.9524072167085813 valid acc 14/16
Epoch 8 loss 0.5518784793515714 valid acc 14/16
Epoch 8 loss 0.7762481897209342 valid acc 15/16
Epoch 8 loss 0.40618942484089365 valid acc 15/16
Epoch 8 loss 0.8117662401004717 valid acc 14/16
Epoch 8 loss 0.5580872252105402 valid acc 15/16
Epoch 8 loss 0.4043468571079657 valid acc 15/16
Epoch 8 loss 0.7972757124051113 valid acc 15/16
Epoch 8 loss 1.3996941553950495 valid acc 15/16
Epoch 8 loss 0.3153346439304958 valid acc 15/16
Epoch 8 loss 0.9629571634392664 valid acc 15/16
Epoch 8 loss 1.4108844220432935 valid acc 15/16
Epoch 8 loss 0.9076118490786501 valid acc 16/16
Epoch 8 loss 0.6110594809909279 valid acc 16/16
Epoch 8 loss 0.6897599307834789 valid acc 16/16
Epoch 8 loss 0.9344996908351444 valid acc 14/16
Epoch 8 loss 0.7845487528246502 valid acc 14/16
Epoch 8 loss 0.7276812553735543 valid acc 14/16
Epoch 8 loss 0.9722355340238917 valid acc 15/16
Epoch 8 loss 1.0254190090571056 valid acc 16/16
Epoch 8 loss 0.3563174099888573 valid acc 16/16
Epoch 8 loss 0.6282445491849302 valid acc 16/16
Epoch 8 loss 0.4931970149172337 valid acc 16/16
Epoch 8 loss 1.326328494345784 valid acc 16/16
Epoch 8 loss 1.2388673544227196 valid acc 16/16
Epoch 8 loss 0.7482696228955377 valid acc 16/16
Epoch 8 loss 0.7163680294341304 valid acc 16/16
Epoch 8 loss 0.38051851855984875 valid acc 16/16
Epoch 8 loss 0.3809272240966109 valid acc 15/16
Epoch 8 loss 1.0285989123598394 valid acc 16/16
Epoch 8 loss 0.43699616138247654 valid acc 16/16
Epoch 8 loss 0.9665014060658892 valid acc 16/16
Epoch 8 loss 0.9189399902972296 valid acc 16/16
Epoch 8 loss 1.0453895384740195 valid acc 16/16
Epoch 8 loss 0.813920318068463 valid acc 16/16
Epoch 8 loss 0.9242877127067559 valid acc 16/16
Epoch 8 loss 0.9234645806951377 valid acc 15/16
Epoch 8 loss 1.2689951910213886 valid acc 15/16
Epoch 8 loss 1.1650764916812695 valid acc 16/16
Epoch 8 loss 1.7057114984426929 valid acc 16/16
Epoch 9 loss 0.07120996111293817 valid acc 15/16
Epoch 9 loss 1.0743865302808249 valid acc 16/16
Epoch 9 loss 1.4981044018535703 valid acc 15/16
Epoch 9 loss 1.1350792732935802 valid acc 14/16
Epoch 9 loss 1.0267170486002732 valid acc 13/16
Epoch 9 loss 1.1770826130212306 valid acc 15/16
Epoch 9 loss 1.3987127881562664 valid acc 15/16
Epoch 9 loss 1.2250173718953639 valid acc 14/16
Epoch 9 loss 1.0386245875408773 valid acc 14/16
Epoch 9 loss 0.8605723502820306 valid acc 14/16
Epoch 9 loss 0.8665741982889243 valid acc 14/16
Epoch 9 loss 1.360405289059785 valid acc 14/16
Epoch 9 loss 1.1773605164910537 valid acc 14/16
Epoch 9 loss 1.0616841887875008 valid acc 14/16
Epoch 9 loss 1.08182057041091 valid acc 14/16
Epoch 9 loss 0.7767600769299914 valid acc 14/16
Epoch 9 loss 1.5168431476870232 valid acc 15/16
Epoch 9 loss 0.6702856763008761 valid acc 15/16
Epoch 9 loss 0.970864164595907 valid acc 15/16
Epoch 9 loss 1.9983178307863767 valid acc 15/16
Epoch 9 loss 3.2044098431881136 valid acc 12/16
Epoch 9 loss 0.4131294624742946 valid acc 13/16
Epoch 9 loss 0.6827306457162832 valid acc 14/16
Epoch 9 loss 1.6189517026151896 valid acc 14/16
Epoch 9 loss 1.2152737140782777 valid acc 14/16
Epoch 9 loss 0.8653677113172821 valid acc 14/16
Epoch 9 loss 0.5774598976287237 valid acc 15/16
Epoch 9 loss 0.4814937502571527 valid acc 14/16
Epoch 9 loss 0.6274486217357742 valid acc 14/16
Epoch 9 loss 0.4443086434837491 valid acc 14/16
Epoch 9 loss 0.5138479933073197 valid acc 15/16
Epoch 9 loss 1.0399865864162947 valid acc 14/16
Epoch 9 loss 0.2783487312549273 valid acc 14/16
Epoch 9 loss 1.170964192856727 valid acc 14/16
Epoch 9 loss 1.2814833933053762 valid acc 15/16
Epoch 9 loss 0.893341352176487 valid acc 14/16
Epoch 9 loss 0.3644893025199534 valid acc 14/16
Epoch 9 loss 0.499177031108523 valid acc 14/16
Epoch 9 loss 0.8123286487425109 valid acc 15/16
Epoch 9 loss 0.9702359100178084 valid acc 14/16
Epoch 9 loss 0.805569965839112 valid acc 14/16
Epoch 9 loss 0.7416795078399949 valid acc 14/16
Epoch 9 loss 1.0727221513426493 valid acc 14/16
Epoch 9 loss 0.9907448647426611 valid acc 14/16
Epoch 9 loss 1.2081881122933056 valid acc 13/16
Epoch 9 loss 0.6403347113939791 valid acc 16/16
Epoch 9 loss 1.044429992620434 valid acc 15/16
Epoch 9 loss 1.0186063420142561 valid acc 16/16
Epoch 9 loss 0.40348651880960296 valid acc 15/16
Epoch 9 loss 0.6548783554435941 valid acc 15/16
Epoch 9 loss 0.7259223204147213 valid acc 15/16
Epoch 9 loss 0.6489995714486579 valid acc 16/16
Epoch 9 loss 1.2022036846895214 valid acc 15/16
Epoch 9 loss 0.6972347070467187 valid acc 14/16
Epoch 9 loss 1.4012896594871078 valid acc 14/16
Epoch 9 loss 0.7683695687482074 valid acc 14/16
Epoch 9 loss 0.6261543315976854 valid acc 15/16
Epoch 9 loss 0.9446684712648441 valid acc 14/16
Epoch 9 loss 0.8165843719924066 valid acc 14/16
Epoch 9 loss 0.6160379682831039 valid acc 14/16
Epoch 9 loss 0.7203498428090995 valid acc 14/16
Epoch 9 loss 1.1308928072821147 valid acc 14/16
Epoch 9 loss 1.0376890060797783 valid acc 14/16
Epoch 10 loss 0.15263743562850235 valid acc 14/16
Epoch 10 loss 0.823477610079627 valid acc 14/16
Epoch 10 loss 1.2073927742647363 valid acc 15/16
Epoch 10 loss 0.7235169507605135 valid acc 14/16
Epoch 10 loss 0.8850547650583734 valid acc 15/16
Epoch 10 loss 0.6206851971836702 valid acc 15/16
Epoch 10 loss 1.187473255720597 valid acc 15/16
Epoch 10 loss 0.5249405023424338 valid acc 15/16
Epoch 10 loss 0.822973968368517 valid acc 15/16
Epoch 10 loss 0.5145830813492286 valid acc 16/16
Epoch 10 loss 0.6864314193073627 valid acc 16/16
Epoch 10 loss 0.9787016234601849 valid acc 16/16
Epoch 10 loss 0.624389658268337 valid acc 16/16
Epoch 10 loss 0.733421273887516 valid acc 15/16
Epoch 10 loss 1.2986277176699925 valid acc 16/16
Epoch 10 loss 1.028216363387097 valid acc 16/16
Epoch 10 loss 1.4253106538236637 valid acc 16/16
Epoch 10 loss 1.0302801876407395 valid acc 16/16
Epoch 10 loss 0.5087713919032286 valid acc 16/16
Epoch 10 loss 0.3987827036905599 valid acc 16/16
Epoch 10 loss 1.1776952135588612 valid acc 14/16
Epoch 10 loss 0.37239855609547134 valid acc 14/16
Epoch 10 loss 0.12406365645534156 valid acc 15/16
Epoch 10 loss 0.39214790309841474 valid acc 15/16
Epoch 10 loss 0.5144867036166219 valid acc 15/16
Epoch 10 loss 0.7454826419129189 valid acc 14/16
Epoch 10 loss 0.2546313517675302 valid acc 15/16
Epoch 10 loss 0.37366864306099884 valid acc 15/16
Epoch 10 loss 0.4544547955777971 valid acc 14/16
Epoch 10 loss 0.3798880464387998 valid acc 14/16
Epoch 10 loss 0.68424473566318 valid acc 15/16
Epoch 10 loss 1.0121555083861518 valid acc 15/16
Epoch 10 loss 0.3896553225138416 valid acc 15/16
Epoch 10 loss 0.5625998887857135 valid acc 14/16
Epoch 10 loss 1.1373071068214646 valid acc 15/16
Epoch 10 loss 0.75486633851728 valid acc 15/16
Epoch 10 loss 0.5740618467774281 valid acc 15/16
Epoch 10 loss 0.42546693353219867 valid acc 16/16
Epoch 10 loss 0.42838015388175377 valid acc 16/16
Epoch 10 loss 0.35576494235543543 valid acc 16/16
Epoch 10 loss 0.31083214359613287 valid acc 16/16
Epoch 10 loss 1.0417511428630304 valid acc 16/16
Epoch 10 loss 0.5063416788836952 valid acc 16/16
Epoch 10 loss 0.5204827331044614 valid acc 15/16
Epoch 10 loss 1.274395932970944 valid acc 15/16
Epoch 10 loss 0.5321172470015214 valid acc 16/16
Epoch 10 loss 1.5647699143926128 valid acc 16/16
Epoch 10 loss 0.6280772297475709 valid acc 16/16
Epoch 10 loss 0.661672097427267 valid acc 16/16
Epoch 10 loss 0.34002268647743095 valid acc 16/16
Epoch 10 loss 0.6971038315009502 valid acc 16/16
Epoch 10 loss 0.5755083917149203 valid acc 16/16
Epoch 10 loss 0.9612850855124594 valid acc 15/16
Epoch 10 loss 0.6799811240714632 valid acc 16/16
Epoch 10 loss 1.3197573113401064 valid acc 15/16
Epoch 10 loss 0.586063728106401 valid acc 16/16
Epoch 10 loss 0.6051267694423986 valid acc 16/16
Epoch 10 loss 0.574591538473377 valid acc 15/16
Epoch 10 loss 1.0089007200287137 valid acc 15/16
Epoch 10 loss 1.1644571289683525 valid acc 15/16
Epoch 10 loss 0.5654565510208623 valid acc 15/16
Epoch 10 loss 1.1008558115371385 valid acc 15/16
Epoch 10 loss 1.3910393765111388 valid acc 16/16