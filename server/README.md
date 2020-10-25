To learning
* !python {...}/Audio-Resolution-with-remove-noise/src/run.py train --train {...}/Audio-Resolution-with-remove-noise/data/vctk/speaker1/vctk-speaker1-train.h5 --val {...}/Audio-Resolution-with-remove-noise/data/vctk/speaker1/vctk-speaker1-val.h5 -e 200 --batch-size 64 --lr 3e-4 --logname {...}/Audio-Resolution-with-remove-noise/src/singlespeaker

To inference
* !python {...}/Audio-Resolution-with-remove-noise/src/run.py eval --logname {...}/Audio-Resolution-with-remove-noise/src/saveFile/model.ckpt-0000 --out-label singlespeaker-out --wav-file-list {...}/Audio-Resolution-with-remove-noise/data/vctk/speaker1/speaker1-val-files.txt --r 4
