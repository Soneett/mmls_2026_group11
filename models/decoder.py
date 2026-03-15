class DotProductDecoder:

    def __call__(self, z_users, z_items):

        return z_users @ z_items.t()
