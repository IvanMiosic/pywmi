(set-logic QF_NRA)
(declare-fun sim_0 () Bool)
(declare-fun sim_1 () Bool)
(declare-fun b_0_0 () Bool)
(declare-fun b_0_1 () Bool)
(declare-fun b_1_0 () Bool)
(declare-fun b_1_1 () Bool)
(declare-fun sim_x () Real)
(declare-fun b_x_0_0 () Real)
(declare-fun b_x_0_1 () Real)
(declare-fun b_x_1_0 () Real)
(declare-fun b_x_1_1 () Real)
(assert (let ((.def_0 (* b_x_1_1 (- 1.0)))) (let ((.def_1 (+ .def_0 1.0))) (let ((.def_2 (ite b_1_1 b_x_1_1 .def_1))) (let ((.def_3 (* b_x_1_0 (- 1.0)))) (let ((.def_4 (+ .def_3 1.0))) (let ((.def_5 (ite b_1_0 b_x_1_0 .def_4))) (let ((.def_6 (* b_x_0_1 (- 1.0)))) (let ((.def_7 (+ .def_6 1.0))) (let ((.def_8 (ite b_0_1 b_x_0_1 .def_7))) (let ((.def_9 (* b_x_0_0 (- 1.0)))) (let ((.def_10 (+ .def_9 1.0))) (let ((.def_11 (ite b_0_0 b_x_0_0 .def_10))) (let ((.def_12 (<= b_x_1_1 1.0))) (let ((.def_13 (ite .def_12 1.0 0.0))) (let ((.def_14 (<= 0.0 b_x_1_1))) (let ((.def_15 (ite .def_14 1.0 0.0))) (let ((.def_16 (* .def_15 .def_13))) (let ((.def_17 (<= b_x_1_0 1.0))) (let ((.def_18 (ite .def_17 1.0 0.0))) (let ((.def_19 (<= 0.0 b_x_1_0))) (let ((.def_20 (ite .def_19 1.0 0.0))) (let ((.def_21 (* .def_20 .def_18))) (let ((.def_22 (<= b_x_0_1 1.0))) (let ((.def_23 (ite .def_22 1.0 0.0))) (let ((.def_24 (<= 0.0 b_x_0_1))) (let ((.def_25 (ite .def_24 1.0 0.0))) (let ((.def_26 (* .def_25 .def_23))) (let ((.def_27 (<= b_x_0_0 1.0))) (let ((.def_28 (ite .def_27 1.0 0.0))) (let ((.def_29 (<= 0.0 b_x_0_0))) (let ((.def_30 (ite .def_29 1.0 0.0))) (let ((.def_31 (* .def_30 .def_28))) (let ((.def_32 (* sim_x (- 1.0)))) (let ((.def_33 (+ .def_32 1.0))) (let ((.def_34 (ite sim_1 sim_x .def_33))) (let ((.def_35 (ite sim_0 sim_x .def_33))) (let ((.def_36 (<= sim_x 1.0))) (let ((.def_37 (ite .def_36 1.0 0.0))) (let ((.def_38 (<= 0.0 sim_x))) (let ((.def_39 (ite .def_38 1.0 0.0))) (let ((.def_40 (* .def_39 .def_37))) (let ((.def_41 (* .def_40 .def_35 .def_34 .def_31 .def_26 .def_21 .def_16 .def_11 .def_8 .def_5 .def_2))) .def_41)))))))))))))))))))))))))))))))))))))))))))
(check-sat)
