#!/bin/bash

set -e

# set TMPDIR variable
export TMPDIR=$_CONDOR_SCRATCH_DIR

echo
echo "I'm running on" $(hostname -f)
echo "OSG site: $OSG_SITE_NAME"
echo

if [[ "$1" != "" ]]; then
    RUN_NAME="$1"
else
    RUN_NAME="not_set"
fi

echo "RUN NAME: $RUN_NAME"

if [[ ! -v OSG_MACHINE_GPUS ]]; then
    echo "OSG_MACHINE_GPUS is not set"
elif [[ -z "$OSG_MACHINE_GPUS" ]]; then
    echo "OSG_MACHINE_GPUS is set to the empty string"
elif [[ $OSG_MACHINE_GPUS == "0" ]]; then
    echo "OSG_MACHINE_GPUS has the value: $OSG_MACHINE_GPUS"
else
    echo "OSG_MACHINE_GPUS has the value: $OSG_MACHINE_GPUS"
    echo "Running nvidia-smi"
    nvidia-smi
fi

python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2628917891
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2628917891
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1970651642
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2144639512
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3280018774
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2899363963
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3379977681
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1799053243
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2273376584
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 919797571
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3567176185
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2947321050
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1842380535
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2424234451
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3826946424
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 4128608742
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2141955573
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 564285217
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1241469302
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3670865299
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 653156764
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 670351107
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1202829223
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2734688537
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3685862192
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 303833536
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3712952546
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2356433740
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2470954468
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 633462137
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1275276983
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 775250968
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1041881446
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2544310540
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 4037159081
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1597415189
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2438374462
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3167620900
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3318117492
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1048015397
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3176812035
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1953386162
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3054004795
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1958728945
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1366716830
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 138023355
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 190340076
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 150467061
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 110475969
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3399158966
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 4063959080
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2810555266
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1230297757
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 999689729
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1428432490
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1824011472
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 882726800
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3716138262
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1741339601
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2743442223
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2436675626
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1873859454
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3571488637
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3140729114
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3809285663
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 4033979505
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 505267026
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1335070893
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1463726299
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3131249045
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2843202851
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1104428358
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 741513725
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 615428238
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1678349647
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2264160777
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3517083455
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2265778277
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 96282594
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2631050111
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3486147453
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 41594665
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2357702987
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1759362420
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2632839886
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3588362432
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1071714595
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2500973280
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 487046729
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3176015417
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3178964385
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1374818387
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2372510381
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2774912113
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 595194256
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2802620950
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 2458691923
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 1507352808
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3498241665
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 4291322289
python ./imbalanced_classification.py --num-runs 1 --run-name $RUN_NAME --seed-val 3802665832

tar zcvf imbalanced_classification.tar.gz imbalanced_classification*
