def get_transform_names_and_params(transforms):
    return [
        "\n".join(
            [
                f"{str(t.__class__).split('.')[-1][:-2]} : {t.pname} = {p}"
                for t, p in zip(transforms.aug_transforms, aug_params)
            ]
        )
        for aug_params in transforms.aug_transform_parameters
    ]
