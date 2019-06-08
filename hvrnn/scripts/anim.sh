#!/bin/sh

USE_ATTENTION=1

if [ -d saved/generated ]; then
    convert saved/generated/img*.png saved/gen_anim.gif
fi

if [ -d saved/predicted_off_0 ]; then
    convert saved/predicted_off_0/img*.png saved/pred_anim_off_0.gif
    if [ $USE_ATTENTION ]; then
        convert saved/predicted_off_0/attention*.png saved/attention_off_0.gif
    fi    
fi

if [ -d saved/predicted_off_1 ]; then
    convert saved/predicted_off_1/img*.png saved/pred_anim_off_1.gif
    if [ $USE_ATTENTION ]; then
        convert saved/predicted_off_1/attention*.png saved/attention_off_1.gif
    fi
fi    

if [ -d saved/predicted_on_10 ]; then
    convert saved/predicted_on_10/img*.png saved/pred_anim_on_10.gif
    if [ $USE_ATTENTION ]; then
        convert saved/predicted_on_10/attention*.png saved/attention_on_10.gif
    fi
fi

if [ -d saved/predicted_on_11 ]; then
    convert saved/predicted_on_11/img*.png saved/pred_anim_on_11.gif
    if [ $USE_ATTENTION ]; then
        convert saved/predicted_on_11/attention*.png saved/attention_on_11.gif
    fi        
fi

if [ -d saved/analyze ]; then
    convert saved/analyze/img00_*.png saved/analyze_anim00.gif
    convert saved/analyze/img01_*.png saved/analyze_anim01.gif
    convert saved/analyze/img10_*.png saved/analyze_anim10.gif
    convert saved/analyze/img11_*.png saved/analyze_anim11.gif

#    convert saved/analyze/analyze_enc_mu0_0.png \
#            saved/analyze/analyze_enc_mu0_1.png \
#            saved/analyze/analyze_enc_mu0_2.png \
#            saved/analyze/analyze_enc_mu0_3.png \
#            saved/analyze/analyze_enc_mu0_4.png \
#            saved/analyze/analyze_enc_mu0_5.png \
#            saved/analyze/analyze_enc_mu0_6.png \
#            saved/analyze/analyze_enc_mu0_7.png \
#            saved/analyze/analyze_enc_mu0_8.png \
#            saved/analyze/analyze_enc_mu0_9.png \
#            saved/analyze_enc_mu0.gif

#    convert saved/analyze/analyze_enc_mu1_0.png \
#            saved/analyze/analyze_enc_mu1_1.png \
#            saved/analyze/analyze_enc_mu1_2.png \
#            saved/analyze/analyze_enc_mu1_3.png \
#            saved/analyze/analyze_enc_mu1_4.png \
#            saved/analyze/analyze_enc_mu1_5.png \
#            saved/analyze/analyze_enc_mu1_6.png \
#            saved/analyze/analyze_enc_mu1_7.png \
#            saved/analyze/analyze_enc_mu1_8.png \
#            saved/analyze/analyze_enc_mu1_9.png \
#            saved/analyze_enc_mu1.gif
    
fi
