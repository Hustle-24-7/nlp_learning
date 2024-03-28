import torch
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=4, nhead=2)
src = torch.rand(2, 3, 4)
out = encoder_layer(src)

transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)
out = transformer_encoder(src)
print(out)

memory = transformer_encoder(src)
decoder_layer = torch.nn.TransformerDecoderLayer(d_model=4, nhead=2)
transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=6)
out_part = torch.rand(2, 3, 4)
out = transformer_decoder(out_part, memory)
print(out)