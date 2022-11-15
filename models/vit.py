from vit_pytorch import ViT

def vit_model(inp_dim, patch_size, n_class, model_type='vit', model_size='base'):
    if model_type == 'vit':
        if model_size == 'base':
            num_layers = 12
            num_heads = 12
            embed_dim = 768
            mlp_dim = 1024
        elif model_size == 'small':
            num_layers = 12
            num_heads = 6
            embed_dim = 384
            mlp_dim = 1024
        elif model_size == 'tiny':
            num_layers = 12
            num_heads = 3
            embed_dim = 192
            mlp_dim = 512
        
        model = ViT(image_size=inp_dim,
                    patch_size=patch_size,
                    num_classes=n_class,
                    dim=embed_dim,
                    depth=num_layers,
                    heads=num_heads,
                    mlp_dim=mlp_dim,
                    channels=1,
                    dropout=0.1,
                    emb_dropout=0.1
                    )
    return model
