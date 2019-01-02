# dialogue-pytorch
Repository for dialogue models which enhance response diversity or coherence, coded in Pytorch.

<p> Contains shared code for multiple concurrent projects. First, this repository contains code for the processing of Reddit
comments for use in training a dialogue model. Second, this repository contains code related to training a baseline
sequence to sequence model (run_seq2seq.py). Finally, run_mismatch.py trains a classifier which diagnoses response coherence with respect
to a dialogue history. With the --mismatch_path flag, the sequence-to-sequence can train using learned coherence
features from the coherence classifier. </p>
