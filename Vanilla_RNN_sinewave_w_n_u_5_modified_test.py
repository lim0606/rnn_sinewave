    [h_t_pred, y_t_pred],_ = theano.scan(fn=recurrent_fn2,
                                         outputs_info=[dict(initial=H_t, taps=[-1]), dict(initial=Y_t, taps=y_t_pred_taps)],
                                        n_steps=500)

    predict2 = theano.function(inputs=[H_t, Y_t],
                               outputs=[h_t_pred, y_t_pred],
                               allow_input_downcast=True)
 

for seq_idx in [1, 10, 100, 200, 300]:
    [h_t_val, y_t_val] = predict1(seq_test[seq_idx,0:50,:])
    y_t_pred0 = np.reshape(np.concatenate((seq_test[seq_idx,50-1,1:n_u], y_t_val[-1])), (n_u,1,1))
    h_t_pred0 = np.reshape(h_t_val[-1], (1,n_h))

    [h_t_pred_val, y_t_pred_val] = predict2(h_t_pred0, y_t_pred0)
    y_t_pred_val = np.reshape(y_t_pred_val, (500, 1))

    y_t_pred_total = np.concatenate((y_t_val, y_t_pred_val), axis=0)

    # We just plot one of the sequences
    plt.close('all')
    fig_pred = plt.figure()

    # Graph 1
    #plt.plot(seq_test[seq_idx,:,:])
    true_targets, = plt.plot(targets_test[seq_idx,:,:], label='y_t, true ouput')
    guessed_targets, = plt.plot(y_t_pred_total, linestyle='--', label='y_t_pred, model output')
    verticalline = plt.axvline(x=49, color='r', linestyle=':')
    plt.grid()
    plt.legend()
    plt.title('true output vs. model output')
    plt.ylim((-3, 3))

    # Save as a file
    plt.savefig('pred_5_seq_idx_%d.png' % seq_idx)
    print "seq_idx: ", seq_idx
