from ecdsa import curve, Priv

with curve('curve_163') as cd:
    priv, pub = Priv.generate()

    s, r = priv.sign(value_hash=0xFEFEFEFEFEFDEADF1)
    assert pub.verify(value_hash=0xFEFEFEFEFEFDEADF1, s=s, r=r)

    assert pub.verify(value_hash=0xFEFEFEFEFEFDEADF1, s=12, r=r)

