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

if [ -e kagglecatsanddogs_5340.tar ]
then
    echo "kagglecatsanddogs_5340.tar found"
    tar xvf kagglecatsanddogs_5340.tar
else
    echo "kagglecatsanddogs_5340.tar not found"
fi

python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2628917891
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2628917891
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1970651642
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2144639512
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3280018774
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2899363963
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3379977681
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1799053243
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2273376584
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 919797571
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3567176185
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2947321050
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1842380535
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2424234451
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3826946424
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 4128608742
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2141955573
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 564285217
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1241469302
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3670865299
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 653156764
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 670351107
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1202829223
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2734688537
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3685862192
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 303833536
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3712952546
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2356433740
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2470954468
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 633462137
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1275276983
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 775250968
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1041881446
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2544310540
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 4037159081
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1597415189
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2438374462
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3167620900
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3318117492
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1048015397
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3176812035
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1953386162
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3054004795
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1958728945
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1366716830
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 138023355
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 190340076
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 150467061
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 110475969
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3399158966
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 4063959080
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2810555266
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1230297757
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 999689729
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1428432490
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1824011472
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 882726800
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3716138262
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1741339601
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2743442223
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2436675626
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1873859454
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3571488637
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3140729114
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3809285663
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 4033979505
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 505267026
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1335070893
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1463726299
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3131249045
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2843202851
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1104428358
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 741513725
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 615428238
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1678349647
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2264160777
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3517083455
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2265778277
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 96282594
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2631050111
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3486147453
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 41594665
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2357702987
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1759362420
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2632839886
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3588362432
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1071714595
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2500973280
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 487046729
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3176015417
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3178964385
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1374818387
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2372510381
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2774912113
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 595194256
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2802620950
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 2458691923
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 1507352808
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3498241665
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 4291322289
python ./image_classification_from_scratch.py --num-runs 1 --run-name $RUN_NAME --seed-val 3802665832

tar zcvf image_classification_from_scratch.tar.gz image_classification_from_scratch*
