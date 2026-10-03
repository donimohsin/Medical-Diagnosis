"""Grad-CAM implemented from scratch for torchvision ResNets (target = last conv block, layer4)."""
import torch

class GradCAM:
    def __init__(self, model, target_layer):
        self.model, self.acts, self.grads = model.eval(), None, None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output.detach()
        # tensor-level hook is safer than module backward hooks with in-place ReLUs
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g.detach()))

    def __call__(self, x, class_idx=None):
        """x: (1,3,H,W) tensor on the model's device. Returns (cam in [0,1] as HxW numpy, class_idx, probs (C,))."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()
            idx = int(logits.argmax(1)) if class_idx is None else int(class_idx)
            logits[0, idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)            # global-average-pooled gradients
        cam = torch.relu((weights * self.acts).sum(dim=1))[0]          # weighted sum of feature maps
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), idx, probs.numpy()
