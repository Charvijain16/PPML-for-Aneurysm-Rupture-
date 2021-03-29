import tensorflow as tf
from tensorpack import TowerTrainer
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper


class GANTrainer(TowerTrainer):

    def __init__(self, model, input_queue):
        super().__init__()
        inputs_desc = model.get_inputs_desc()

        # Setup input
        cbs = input_queue.setup(inputs_desc)
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, inputs_desc)
        with TowerContext('', is_training=True):
            self.tower_func(*input_queue.get_input_tensors())

        opt = model.get_optimizer()
        opt_disc = model.get_disc_optimizer()
        # Define the training iteration by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            g_min_grad = opt.compute_gradients(model.g_loss, var_list=model.g_vars)
            g_min_grad_clip = [
                (tf.clip_by_value(grad, -5.0, 5.0), var)
                for grad, var in g_min_grad
            ]

            g_min_train_op = opt.apply_gradients(g_min_grad_clip, name='g_op')
            with tf.control_dependencies([g_min_train_op]):
                d_min_grad = opt_disc.compute_gradients(model.d_loss, var_list=model.d_vars)
                d_min_grad_clip = [
                    (tf.clip_by_value(grad, -5.0, 5.0), var)
                    for grad, var in d_min_grad
                ]
                d_min_train_op = opt_disc.apply_gradients(d_min_grad_clip, name='d_op')

        self.train_op = d_min_train_op
