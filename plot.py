from rnn_jhlim import *

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

if __name__ == '__main__':
    encoder = 'lstm'
    optimizer=adadelta
    initializer=ortho_weight

    # plot predpred err
    plt.close('all')
    fig_pred = plt.figure(1)
    plt.grid()
    plt.title('prediction error per step (%s_%s_%s)' % (encoder, optimizer.__name__, initializer.__name__))


    fig_pred_per_frame = plt.figure(2)
    plt.grid()
    plt.title('prediction error per frame (%s_%s_%s)' % (encoder, optimizer.__name__, initializer.__name__))
    

    for noise_std in [0., 0.001]:
        for dim_proj in [10, 100]:
            for input_dim in [1, 5]:
                
                #noise_std = 0.#0.001
                #dim_proj = 10
                #input_dim=5
                output_dim=input_dim # 1
                stride=input_dim # 1
                pred_steps = ((160-(input_dim+output_dim))/stride+1) - ((50-input_dim)/stride+1)
                #encoder = 'lstm'
                mode = 'test'
                #max_epochs=800
                #optimizer=adadelta
                #initializer=ortho_weight
                filename="models/0model_%s_%s_%s_input_dim_%d_hidden_%d_noise_std_%.4f.npz" % (encoder, optimizer.__name__, initializer.__name__, input_dim, dim_proj, noise_std)

                is_plot_save=False

                # test
                reload_model=filename
                saveto=""
                max_epochs=0

                train_err, valid_err, test_err, pred_err, pred_err_per_frame = train_lstm(
                        dim_proj=dim_proj,
                        input_dim=input_dim,
                        output_dim=output_dim,
                        stride=stride,
                        pred_steps=pred_steps,
                        reload_model=reload_model,
                        saveto=saveto,
                        noise_std=noise_std,
                        max_epochs=max_epochs,
                        test_size=500,
                        use_dropout=False,
                        encoder=encoder,
                        initializer=initializer,
                        optimizer=optimizer,
                        is_plot_save=is_plot_save
                )            
                plt.figure(1) # fig_pred
                plot_pred_err, = plt.plot(pred_err, '--o', label='i=%d h=%d n_std=%.4f' % (input_dim, dim_proj, noise_std))
 
                plt.figure(2) #fig_pred_per_frame
                plot_pred_err_per_frame, = plt.plot(pred_err_per_frame, '--o', label='i=%d h=%d n_std=%.4f' % (input_dim, dim_proj, noise_std))

    # Save as a file
    plt.figure(1)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('mse')
    plt.savefig('0pred_err_%s_%s_%s.png' % (encoder, optimizer.__name__, initializer.__name__), bbox_extra_artists=(lgd,), bbox_inches='tight')
    

    plt.figure(1)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.yscale('log')
    plt.xlim((0, 20))
    plt.xlabel('step')
    plt.ylabel('mse')
    plt.savefig('0pred_err_%s_%s_%s_zoom.png' % (encoder, optimizer.__name__, initializer.__name__), bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(2)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.yscale('log')
    plt.xlabel('k')
    plt.ylabel('mse')
    plt.savefig('0pred_err_per_frame_%s_%s_%s.png' % (encoder, optimizer.__name__, initializer.__name__), bbox_extra_artists=(lgd,), bbox_inches='tight')

    plt.figure(2)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.yscale('log')
    plt.xlim((0, 20))
    plt.xlabel('k')
    plt.ylabel('mse')
    plt.savefig('0pred_err_per_frame_%s_%s_%s_zoom.png' % (encoder, optimizer.__name__, initializer.__name__), bbox_extra_artists=(lgd,), bbox_inches='tight')

