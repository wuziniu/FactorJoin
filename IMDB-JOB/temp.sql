SELECT COUNT(*) FROM A, B, C, D, E, F WHERE 
 A.id = B.Aid AND A.id = E.Aid AND
 A.id2 = B.Aid2 AND A.id2 = C.Aid2 AND
 C.id = D.Cid AND
 E.id = D.Eid AND E.id = F.Eid AND D.Eid = F.Eid
 AND Q(A) AND Q(B) AND Q(C) AND Q(D) AND Q(E)
 AND Q(F);

