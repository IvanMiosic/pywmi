(set-logic QF_NRA)
(declare-fun x0 () Real)
(declare-fun x1 () Real)
(declare-fun x2 () Real)
(declare-fun x3 () Real)
(declare-fun x4 () Real)
(declare-fun x5 () Real)
(declare-fun x6 () Real)
(declare-fun x7 () Real)
(assert (let ((.def_0 (* x7 (- 1.0)))) (let ((.def_1 (* .def_0 x7))) (let ((.def_2 (+ .def_1 1.0))) (let ((.def_3 (+ x7 1.0))) (let ((.def_4 (< x7 0.0))) (let ((.def_5 (ite .def_4 .def_3 .def_2))) (let ((.def_6 (< x7 1.0))) (let ((.def_7 (< (- 1.0) x7))) (let ((.def_8 (and .def_7 .def_6))) (let ((.def_9 (ite .def_8 .def_5 0.0))) (let ((.def_10 (* x6 (- 1.0)))) (let ((.def_11 (* .def_10 x6))) (let ((.def_12 (+ .def_11 1.0))) (let ((.def_13 (+ x6 1.0))) (let ((.def_14 (< x6 0.0))) (let ((.def_15 (ite .def_14 .def_13 .def_12))) (let ((.def_16 (< x6 1.0))) (let ((.def_17 (< (- 1.0) x6))) (let ((.def_18 (and .def_17 .def_16))) (let ((.def_19 (ite .def_18 .def_15 0.0))) (let ((.def_20 (* x5 (- 1.0)))) (let ((.def_21 (* .def_20 x5))) (let ((.def_22 (+ .def_21 1.0))) (let ((.def_23 (+ x5 1.0))) (let ((.def_24 (< x5 0.0))) (let ((.def_25 (ite .def_24 .def_23 .def_22))) (let ((.def_26 (< x5 1.0))) (let ((.def_27 (< (- 1.0) x5))) (let ((.def_28 (and .def_27 .def_26))) (let ((.def_29 (ite .def_28 .def_25 0.0))) (let ((.def_30 (* x4 (- 1.0)))) (let ((.def_31 (* .def_30 x4))) (let ((.def_32 (+ .def_31 1.0))) (let ((.def_33 (+ x4 1.0))) (let ((.def_34 (< x4 0.0))) (let ((.def_35 (ite .def_34 .def_33 .def_32))) (let ((.def_36 (< x4 1.0))) (let ((.def_37 (< (- 1.0) x4))) (let ((.def_38 (and .def_37 .def_36))) (let ((.def_39 (ite .def_38 .def_35 0.0))) (let ((.def_40 (* x3 (- 1.0)))) (let ((.def_41 (* .def_40 x3))) (let ((.def_42 (+ .def_41 1.0))) (let ((.def_43 (+ x3 1.0))) (let ((.def_44 (< x3 0.0))) (let ((.def_45 (ite .def_44 .def_43 .def_42))) (let ((.def_46 (< x3 1.0))) (let ((.def_47 (< (- 1.0) x3))) (let ((.def_48 (and .def_47 .def_46))) (let ((.def_49 (ite .def_48 .def_45 0.0))) (let ((.def_50 (* x2 (- 1.0)))) (let ((.def_51 (* .def_50 x2))) (let ((.def_52 (+ .def_51 1.0))) (let ((.def_53 (+ x2 1.0))) (let ((.def_54 (< x2 0.0))) (let ((.def_55 (ite .def_54 .def_53 .def_52))) (let ((.def_56 (< x2 1.0))) (let ((.def_57 (< (- 1.0) x2))) (let ((.def_58 (and .def_57 .def_56))) (let ((.def_59 (ite .def_58 .def_55 0.0))) (let ((.def_60 (* x1 (- 1.0)))) (let ((.def_61 (* .def_60 x1))) (let ((.def_62 (+ .def_61 1.0))) (let ((.def_63 (+ x1 1.0))) (let ((.def_64 (< x1 0.0))) (let ((.def_65 (ite .def_64 .def_63 .def_62))) (let ((.def_66 (< x1 1.0))) (let ((.def_67 (< (- 1.0) x1))) (let ((.def_68 (and .def_67 .def_66))) (let ((.def_69 (ite .def_68 .def_65 0.0))) (let ((.def_70 (* x0 (- 1.0)))) (let ((.def_71 (* .def_70 x0))) (let ((.def_72 (+ .def_71 1.0))) (let ((.def_73 (+ x0 1.0))) (let ((.def_74 (< x0 0.0))) (let ((.def_75 (ite .def_74 .def_73 .def_72))) (let ((.def_76 (< x0 1.0))) (let ((.def_77 (< (- 1.0) x0))) (let ((.def_78 (and .def_77 .def_76))) (let ((.def_79 (ite .def_78 .def_75 0.0))) (let ((.def_80 (* .def_79 .def_69 .def_59 .def_49 .def_39 .def_29 .def_19 .def_9))) .def_80))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
(check-sat)