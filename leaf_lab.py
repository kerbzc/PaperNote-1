# leaf_lab.py — Week 5 teaching simulation: "What did the model actually learn?"
# Everything here is SYNTHETIC. It is a toy world where we know the truth, so we can test our reasoning.
#   python3 leaf_lab.py            -> trains the model, prints the five accuracies from the lecture
#   python3 leaf_lab.py --mine     -> also runs YOUR intervention (edit my_intervention below)
import sys, numpy as np, cv2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from leafgen import make_all, feats, background      # feats = what the model 'sees': an HSV colour histogram (8 x 4 x 4 bins)

D = make_all(seed=5)                                   # generated on the fly (about 5 s); same images as the lecture
Xtr, ytr, Xte, yte, Xf, yf = D["Xtr"], D["ytr"], D["Xte"], D["yte"], D["Xf"], D["yf"]
masks, leaves = D["masks"], D["leaves"]                # for the lab TEST set: leaf mask and leaf-only layer

def train(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
    clf.fit(feats(X), y)
    return clf

def acc(clf, X, y):
    return (clf.predict(feats(X)) == y).mean()

def recomposite(im, mask, leaf, bg_value, rng):
    """Put the SAME leaf on a new uniform background, through the same editing steps (erode 1 px, add noise)."""
    m = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
    bg = np.full_like(im, bg_value)
    out = np.where(m[:, :, None], leaf, bg)
    return np.clip(out + rng.normal(0, 6, out.shape), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    clf = train(Xtr, ytr)
    print(f"lab test  (same protocol)      {acc(clf, Xte, yte):.3f}")
    print(f"field     (new protocol)       {acc(clf, Xf, yf):.3f}")
    # intervention: swap the background type (healthy -> dark board, diseased -> light board)
    Xswap = np.array([recomposite(im, m, l, 205 if d else 135, rng) for im, m, l, d in zip(Xte, masks, leaves, yte)])
    Xsham = np.array([recomposite(im, m, l, 135 if d else 205, rng) for im, m, l, d in zip(Xte, masks, leaves, yte)])
    print(f"background swapped             {acc(clf, Xswap, yte):.3f}")
    print(f"sham edit (same background)    {acc(clf, Xsham, yte):.3f}")
    Xleaf = np.array([np.where((m > 0)[:, :, None], im, 0).astype(np.uint8) for im, m in zip(Xte, masks)])
    print(f"leaf only (background black)   {acc(clf, Xleaf, yte):.3f}")

    if "--mine" in sys.argv:
        print("\n========================================")
        print("\nProfessor's Intervention: Brightness +40")
        # ---------------------------------------------------------------- YOUR intervention
        # Change ONE thing about the lab test images and see what the model does.
        # Ideas: brightness of the whole image; blur; remove the brown spots (paint them green);
        #        rotate the leaf; put the leaf on a FIELD background (use Xf backgrounds).
        # Write down: (1) what you changed, (2) your prediction BEFORE running, (3) the number, (4) one alternative explanation.
        def my_intervention(im, mask, leaf, label):
            out = im.copy()
            out = np.clip(out.astype(int) + 40, 0, 255).astype(np.uint8)     # example: brighter by 40 (edit me)
            return out
        Xmine = np.array([my_intervention(im, m, l, d) for im, m, l, d in zip(Xte, masks, leaves, yte)])
        print(f"• MY intervention                {acc(clf, Xmine, yte):.3f}")



    
    #================================================================================
    #MY INTERVENTION: I will blur the whole image using a 7 x 7 gaussian filter. 

    #======================PREDICTION USING MY INTERVENTION==========================
    #Prediction: Accuracy may drop because blur makes leaf details less clear.
    #Alternative: Accuracy may stay high because the model can still use colours.
    
    if "--mine2" in sys.argv:
        print("\nImage Change Test: Blur 7 X 7")

        def my_intervention(im, mask, leaf, label):
            out = im.copy()
            out = cv2.GaussianBlur(out, (7, 7), 0) #One change: Blur the whole image using a 7 x 7 gaussian filter
            return out

        Xmine = np.array([my_intervention(im, m, l, d)
            for im, m, l, d in zip(Xte, masks, leaves, yte)])

        print(f"• Blur (7 x 7)                  {acc(clf, Xmine, yte):.3f}")

    #My Additional Intervention: Make the original images darker
    #Prediction: Accuracy may drop because the colours become darker.
    #Alternative: Accuracy may stay high if useful colour differences remain.

    if "--mine3" in sys.argv:
        print("\nImage Change Test: Darker Images")
        
        def darker_images(im):
            out = im.copy()
            out = np.clip(out.astype(int) - 40, 0, 255).astype(np.uint8)
            return out

        Xdark = np.array([darker_images(im) for im in Xte])

        print(f"• Brightness -40                {acc(clf, Xdark, yte):.3f}\n")
        print("========================================")
