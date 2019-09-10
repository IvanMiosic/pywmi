(set-logic QF_LRA)
(declare-fun x0 () Real)
(assert (let ((.def_0 (* 2.0 x0))) (let ((.def_1 (+ .def_0 1.0))) .def_1)))
(check-sat)
