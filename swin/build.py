from .swin_transformer import SwinTransformer


def build_model(args):

    model = SwinTransformer(
                img_size= 384,
                patch_size= 4,
                in_chans= 3,
                embed_dim= 128,
                depths= [2, 2, 18, 2],
                num_heads= [4, 8, 16, 32],
                window_size= 12,
                mlp_ratio= 4.0,
                qkv_bias= True,
                qk_scale= None,
                drop_rate= 0.0,
                drop_path_rate= 0.5,
                ape= False,
                patch_norm= True,
                args = args
        )
    return model
