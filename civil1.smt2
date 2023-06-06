(declare-sort Person)

(declare-fun name (Person) String)
(declare-fun age (Person) Int)
(declare-fun gender (Person) String)
(declare-fun marital_status (Person) String)

(assert (forall ((p1 Person) (p2 Person))
  (implies (and (= (name p1) (name p2)) (not (= p1 p2)))
    false)))

(assert (forall ((p Person))
  (or (= (marital_status p) "single")
      (= (marital_status p) "married")
      (= (marital_status p) "divorced")
      (= (marital_status p) "widowed"))))

(assert (forall ((p Person))
  (>= (age p) 0)))

(assert (forall ((p Person))
  (or (= (gender p) "male") (= (gender p) "female"))))

(check-sat)
(get-model)
