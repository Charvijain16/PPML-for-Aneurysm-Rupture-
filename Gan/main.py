from experiment import run_experiment

if __name__ == '__main__':
    run_experiment(name="Aneurysm_HParameters_DiffPrivate10",
                   steps_per_epoch=100,
                   sample_rows=450,
                   train_csv="../Preprocessing/preprocessed_data1.csv",
                   continuous_cols=[2, 5, 15, 16, 21, 23, 24, 26, 29, 30, 31, 37],
                   num_random_search=1)
    # model = TGANModel.load("/home/charvijain16/Desktop/PrivacyPreservingSensitiveData/Gan/experiments_2502/Aneurysm_HParameters_DiffPrivate2/trained_model_11")
    # print(model.batch_size,
    #       model.z_dim,
    #       model.noise,
    #       model.l2norm,
    #       model.learning_rate,
    #       model.num_gen_rnn,
    #       model.num_gen_feature,
    #       model.num_dis_layers,
    #       model.num_dis_hidden,
    #       model.optimizer,
    #       model.disc_optimizer,
    #       model.microbatches,
    #       model.noise_multiplier,
    #       model.l2_norm_clip)
    #
