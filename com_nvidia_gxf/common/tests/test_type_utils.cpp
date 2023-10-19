/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "common/type_utils.hpp"

namespace nvidia {

static_assert( IsSame<void, void>::value);
static_assert(!IsSame<const void, void>::value);
static_assert(!IsSame<unsigned int, int>::value);
static_assert( IsSame_v<void, void>);

static_assert( IsSame<void_t<>, void>::value);
static_assert( IsSame<void_t<struct A>, void>::value);

static_assert( IsSame<TypeIdentity<int>::type, int>::value);
static_assert(!IsSame<TypeIdentity<int*>::type, int>::value);
static_assert( IsSame<TypeIdentity<int*>::type, TypeIdentity_t<int*>>::value);

static_assert( IntegralConstant<int, 5>::value == 5);
static_assert( IntegralConstant<bool, true>::value);
static_assert(!IntegralConstant<bool, false>::value);

static_assert( BoolConstant<true>::value);
static_assert(!BoolConstant<false>::value);

static_assert( TrueType::value);
static_assert(!FalseType::value);

static_assert( IsSame<Conditional<true, int, char>::type, int>::value);
static_assert(!IsSame<Conditional<true, int, char>::type, char>::value);
static_assert(!IsSame<Conditional<false, int, char>::type, int>::value);
static_assert( IsSame<Conditional<false, int, char>::type, char>::value);
static_assert( IsSame<Conditional<true, int, char>::type,
                      Conditional_t<true, int, char>>::value);

static_assert( Conjunction<>::value);
static_assert( Conjunction<TrueType>::value);
static_assert(!Conjunction<FalseType>::value);
static_assert( Conjunction<TrueType, TrueType>::value);
static_assert(!Conjunction<TrueType, FalseType>::value);
static_assert(!Conjunction<FalseType, TrueType>::value);
static_assert(!Conjunction<FalseType, FalseType>::value);
static_assert( Conjunction<>::value == Conjunction_v<>);

static_assert(!Disjunction<>::value);
static_assert( Disjunction<TrueType>::value);
static_assert(!Disjunction<FalseType>::value);
static_assert( Disjunction<TrueType, TrueType>::value);
static_assert( Disjunction<TrueType, FalseType>::value);
static_assert( Disjunction<FalseType, TrueType>::value);
static_assert(!Disjunction<FalseType, FalseType>::value);
static_assert( Disjunction<>::value == Disjunction_v<>);

static_assert(!Negation<TrueType>::value);
static_assert( Negation<FalseType>::value);
static_assert( Negation<TrueType>::value == Negation_v<TrueType>);

static_assert( IsSame<RemoveReference<int>::type, int>::value);
static_assert( IsSame<RemoveReference<int&>::type, int>::value);
static_assert( IsSame<RemoveReference<int&&>::type, int>::value);
static_assert( IsSame<RemoveReference<int*>::type, int*>::value);
static_assert( IsSame<RemoveReference<int*&>::type, int*>::value);
static_assert( IsSame<RemoveReference<int*&&>::type, int*>::value);
static_assert( IsSame<RemoveReference<int&&>::type, RemoveReference_t<int&&>>::value);

static_assert( IsSame<RemoveCV<int>::type, int>::value);
static_assert( IsSame<RemoveCV<const void>::type, void>::value);
static_assert( IsSame<RemoveCV<const int>::type, int>::value);
static_assert( IsSame<RemoveCV<volatile void>::type, void>::value);
static_assert( IsSame<RemoveCV<volatile int>::type, int>::value);
static_assert( IsSame<RemoveCV<const volatile void>::type, void>::value);
static_assert( IsSame<RemoveCV<const volatile int>::type, int>::value);
static_assert( IsSame<RemoveCV<volatile const int>::type, int>::value);
static_assert( IsSame<RemoveCV<const volatile int>::type,
                      RemoveCV_t<const volatile int>>::value);

static_assert( IsSame<RemoveConst<int>::type, int>::value);
static_assert( IsSame<RemoveConst<const int>::type, int>::value);
static_assert( IsSame<RemoveConst<volatile int>::type, volatile int>::value);
static_assert( IsSame<RemoveConst<const volatile int>::type, volatile int>::value);
static_assert( IsSame<RemoveConst<volatile const int>::type, volatile int>::value);
static_assert( IsSame<RemoveConst<const volatile int>::type,
                      RemoveConst_t<const volatile int>>::value);

static_assert( IsSame<RemoveVolatile<int>::type, int>::value);
static_assert( IsSame<RemoveVolatile<const int>::type, const int>::value);
static_assert( IsSame<RemoveVolatile<volatile int>::type, int>::value);
static_assert( IsSame<RemoveVolatile<const volatile int>::type, const int>::value);
static_assert( IsSame<RemoveVolatile<volatile const int>::type, const int>::value);
static_assert( IsSame<RemoveVolatile<const volatile int>::type,
                      RemoveVolatile_t<const volatile int>>::value);

static_assert( IsSame<RemoveCVRef<int>::type, int>::value);
static_assert( IsSame<RemoveCVRef<int&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<int&&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const int>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const int&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const int&&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<volatile int>::type, int>::value);
static_assert( IsSame<RemoveCVRef<volatile int&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<volatile int&&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const volatile int>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const volatile int&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const volatile int&&>::type, int>::value);
static_assert( IsSame<RemoveCVRef<const volatile int&>::type,
                      RemoveCVRef_t<const volatile int>>::value);

static_assert( IsSame<AddLvalueReference<int>::type, int&>::value);
static_assert( IsSame<AddLvalueReference<int&>::type, int&>::value);
static_assert( IsSame<AddLvalueReference<int&&>::type, int&>::value);
static_assert( IsSame<AddLvalueReference<void>::type, void>::value);
static_assert( IsSame<AddLvalueReference<const void>::type, const void>::value);
static_assert( IsSame<AddLvalueReference<const volatile void>::type, const volatile void>::value);
static_assert( IsSame<AddLvalueReference<int>::type, AddLvalueReference_t<int>>::value);

static_assert( IsSame<AddRvalueReference<int>::type, int&&>::value);
static_assert( IsSame<AddRvalueReference<int&>::type, int&>::value);
static_assert( IsSame<AddRvalueReference<int&&>::type, int&&>::value);
static_assert( IsSame<AddRvalueReference<void>::type, void>::value);
static_assert( IsSame<AddRvalueReference<const void>::type, const void>::value);
static_assert( IsSame<AddRvalueReference<const volatile void>::type, const volatile void>::value);
static_assert( IsSame<AddRvalueReference<int>::type, AddRvalueReference_t<int>>::value);

static_assert( IsSame<decltype(Declval<int>()), int&&>::value);

static_assert( IsSame<Decay<int>::type, int>::value);
static_assert( IsSame<Decay<int*>::type, int*>::value);
static_assert( IsSame<Decay<int&>::type, int>::value);
static_assert( IsSame<Decay<int&&>::type, int>::value);
static_assert( IsSame<Decay<const int&>::type, int>::value);
static_assert( IsSame<Decay<int[2]>::type, int*>::value);
static_assert( IsSame<Decay<int(int)>::type, int(*)(int)>::value);
static_assert( IsSame<Decay<void>::type, void>::value);

template <bool, class = void>
struct TestEnableIf : FalseType {};

template <bool B>
struct TestEnableIf<B, typename EnableIf<B>::type> : TrueType {
  using EnableIfType = EnableIf_t<B, int*>;
};

static_assert( TestEnableIf<true>::value);
static_assert(!TestEnableIf<false>::value);
static_assert( IsSame<TestEnableIf<true>::EnableIfType, int*>::value);

static_assert(!IsReference<int>::value);
static_assert(!IsReference<int*>::value);
static_assert( IsReference<int&>::value);
static_assert( IsReference<int&&>::value);
static_assert( IsReference<int*&>::value);
static_assert(!IsReference_v<int>);

static_assert( IsConstructible<int, int&>::value);
static_assert(!IsConstructible<int, int*>::value);
static_assert( IsConstructible<int*, int*>::value);
static_assert(!IsConstructible<int*, int&>::value);
static_assert( IsConstructible_v<int, int&>);

struct A {
  A() noexcept(true) {}
  A(int) noexcept(true) {}
  explicit A(float*) noexcept(true) {}
};

struct B {
  B() noexcept(false) {}
  B(int) noexcept(false) {}
  explicit B(float*) noexcept(false) {}
};

static_assert( IsNothrowConstructible<A, int>::value);
static_assert(!IsNothrowConstructible<B, int>::value);
static_assert(!IsNothrowConstructible<A, B>::value);
static_assert(!IsNothrowConstructible<B, A>::value);
static_assert( IsNothrowConstructible<int*, int*>::value);
static_assert(!IsNothrowConstructible<int*, int&>::value);
static_assert( IsNothrowConstructible_v<A, int>);
static_assert(!IsNothrowConstructible_v<B, int>);
static_assert(!IsNothrowConstructible_v<A, B>);
static_assert(!IsNothrowConstructible_v<B, A>);

struct C {
  C() = delete;
};

static_assert( IsDefaultConstructible<A>::value);
static_assert( IsDefaultConstructible<B>::value);
static_assert(!IsDefaultConstructible<C>::value);
static_assert( IsDefaultConstructible_v<A>);
static_assert( IsDefaultConstructible_v<B>);
static_assert(!IsDefaultConstructible_v<C>);

static_assert( IsNothrowDefaultConstructible<A>::value);
static_assert(!IsNothrowDefaultConstructible<B>::value);
static_assert(!IsNothrowDefaultConstructible<C>::value);
static_assert( IsNothrowDefaultConstructible_v<A>);
static_assert(!IsNothrowDefaultConstructible_v<B>);
static_assert(!IsNothrowDefaultConstructible_v<C>);

static_assert( IsNothrowConvertible<void, void>::value);
static_assert( IsNothrowConvertible<const void, void>::value);
static_assert( IsNothrowConvertible<void, const void>::value);
static_assert( IsNothrowConvertible<int, A>::value);
static_assert(!IsNothrowConvertible<float*, A>::value);
static_assert( IsNothrowConvertible_v<int, A>);
static_assert(!IsNothrowConvertible<int, B>::value);
static_assert(!IsNothrowConvertible<float*, B>::value);
static_assert(!IsNothrowConvertible_v<int, B>);

static_assert( IsConvertible<void, void>::value);
static_assert( IsConvertible<const void, void>::value);
static_assert( IsConvertible<void, const void>::value);
static_assert( IsConvertible<int, A>::value);
static_assert(!IsConvertible<float*, A>::value);
static_assert( IsConvertible_v<int, A>);
static_assert( IsConvertible<int, B>::value);
static_assert(!IsConvertible<float*, B>::value);
static_assert( IsConvertible_v<int, B>);

static_assert( IsAssignable<int&, int>::value);
static_assert( IsAssignable<int&, char>::value);
static_assert(!IsAssignable<int&, int*>::value);
static_assert( IsAssignable_v<int&, int&>);

static_assert( IsVoid<void>::value);
static_assert( IsVoid<const void>::value);
static_assert( IsVoid<volatile void>::value);
static_assert( IsVoid<const volatile void>::value);
static_assert(!IsVoid<void*>::value);
static_assert( IsVoid_v<void>);

static_assert(!IsConst<int>::value);
static_assert( IsConst<const int>::value);
static_assert(!IsConst<volatile int>::value);
static_assert( IsConst<volatile const int>::value);
static_assert(!IsConst<void>::value);
static_assert( IsConst<const void>::value);
static_assert(!IsConst<const int*>::value);
static_assert( IsConst<int* const>::value);
static_assert(!IsConst<const int&>::value);
static_assert(!IsConst_v<int>);
static_assert( IsConst_v<const int>);

static_assert( IsIntegral<bool>::value);
static_assert( IsIntegral<int8_t>::value);
static_assert( IsIntegral<uint8_t>::value);
static_assert( IsIntegral<int16_t>::value);
static_assert( IsIntegral<uint16_t>::value);
static_assert( IsIntegral<int32_t>::value);
static_assert( IsIntegral<uint32_t>::value);
static_assert( IsIntegral<int64_t>::value);
static_assert( IsIntegral<uint64_t>::value);
static_assert(!IsIntegral<float>::value);
static_assert(!IsIntegral<double>::value);
static_assert(!IsIntegral<void>::value);
static_assert( IsIntegral_v<bool>);
static_assert( IsIntegral_v<int8_t>);
static_assert( IsIntegral_v<uint8_t>);
static_assert( IsIntegral_v<int16_t>);
static_assert( IsIntegral_v<uint16_t>);
static_assert( IsIntegral_v<int32_t>);
static_assert( IsIntegral_v<uint32_t>);
static_assert( IsIntegral_v<int64_t>);
static_assert( IsIntegral_v<uint64_t>);
static_assert(!IsIntegral_v<float>);
static_assert(!IsIntegral_v<double>);
static_assert(!IsIntegral_v<void>);

static_assert( IsFloatingPoint<float>::value);
static_assert( IsFloatingPoint<double>::value);
static_assert( IsFloatingPoint<long double>::value);
static_assert(!IsFloatingPoint<int>::value);
static_assert(!IsFloatingPoint<bool>::value);
static_assert(!IsFloatingPoint<void>::value);
static_assert( IsFloatingPoint_v<float>);
static_assert( IsFloatingPoint_v<double>);
static_assert( IsFloatingPoint_v<long double>);
static_assert(!IsFloatingPoint_v<int>);
static_assert(!IsFloatingPoint_v<bool>);
static_assert(!IsFloatingPoint_v<void>);

static_assert( IsArithmetic<bool>::value);
static_assert( IsArithmetic<int8_t>::value);
static_assert( IsArithmetic<uint8_t>::value);
static_assert( IsArithmetic<int16_t>::value);
static_assert( IsArithmetic<uint16_t>::value);
static_assert( IsArithmetic<int32_t>::value);
static_assert( IsArithmetic<uint32_t>::value);
static_assert( IsArithmetic<int64_t>::value);
static_assert( IsArithmetic<uint64_t>::value);
static_assert( IsArithmetic<float>::value);
static_assert( IsArithmetic<double>::value);
static_assert( IsArithmetic<long double>::value);
static_assert(!IsArithmetic<void>::value);
static_assert( IsArithmetic_v<bool>);
static_assert( IsArithmetic_v<int8_t>);
static_assert( IsArithmetic_v<uint8_t>);
static_assert( IsArithmetic_v<int16_t>);
static_assert( IsArithmetic_v<uint16_t>);
static_assert( IsArithmetic_v<int32_t>);
static_assert( IsArithmetic_v<uint32_t>);
static_assert( IsArithmetic_v<int64_t>);
static_assert( IsArithmetic_v<uint64_t>);
static_assert( IsArithmetic_v<float>);
static_assert( IsArithmetic_v<double>);
static_assert( IsArithmetic_v<long double>);
static_assert(!IsArithmetic_v<void>);

static_assert(!IsSigned<bool>::value);
static_assert( IsSigned<int8_t>::value);
static_assert(!IsSigned<uint8_t>::value);
static_assert( IsSigned<int16_t>::value);
static_assert(!IsSigned<uint16_t>::value);
static_assert( IsSigned<int32_t>::value);
static_assert(!IsSigned<uint32_t>::value);
static_assert( IsSigned<int64_t>::value);
static_assert(!IsSigned<uint64_t>::value);
static_assert( IsSigned<float>::value);
static_assert( IsSigned<double>::value);
static_assert( IsSigned<long double>::value);
static_assert(!IsSigned<void>::value);
static_assert(!IsSigned_v<bool>);
static_assert( IsSigned_v<int8_t>);
static_assert(!IsSigned_v<uint8_t>);
static_assert( IsSigned_v<int16_t>);
static_assert(!IsSigned_v<uint16_t>);
static_assert( IsSigned_v<int32_t>);
static_assert(!IsSigned_v<uint32_t>);
static_assert( IsSigned_v<int64_t>);
static_assert(!IsSigned_v<uint64_t>);
static_assert( IsSigned_v<float>);
static_assert( IsSigned_v<double>);
static_assert( IsSigned_v<long double>);
static_assert(!IsSigned_v<void>);

static_assert( IsUnsigned<bool>::value);
static_assert(!IsUnsigned<int8_t>::value);
static_assert( IsUnsigned<uint8_t>::value);
static_assert(!IsUnsigned<int16_t>::value);
static_assert( IsUnsigned<uint16_t>::value);
static_assert(!IsUnsigned<int32_t>::value);
static_assert( IsUnsigned<uint32_t>::value);
static_assert(!IsUnsigned<int64_t>::value);
static_assert( IsUnsigned<uint64_t>::value);
static_assert(!IsUnsigned<float>::value);
static_assert(!IsUnsigned<double>::value);
static_assert(!IsUnsigned<long double>::value);
static_assert(!IsUnsigned<void>::value);
static_assert( IsUnsigned_v<bool>);
static_assert(!IsUnsigned_v<int8_t>);
static_assert( IsUnsigned_v<uint8_t>);
static_assert(!IsUnsigned_v<int16_t>);
static_assert( IsUnsigned_v<uint16_t>);
static_assert(!IsUnsigned_v<int32_t>);
static_assert( IsUnsigned_v<uint32_t>);
static_assert(!IsUnsigned_v<int64_t>);
static_assert( IsUnsigned_v<uint64_t>);
static_assert(!IsUnsigned_v<float>);
static_assert(!IsUnsigned_v<double>);
static_assert(!IsUnsigned_v<long double>);
static_assert(!IsUnsigned_v<void>);

static_assert(IsSame_v<MakeSigned<int8_t>::type,   int8_t>);
static_assert(IsSame_v<MakeSigned<uint8_t>::type,  int8_t>);
static_assert(IsSame_v<MakeSigned<int16_t>::type,  int16_t>);
static_assert(IsSame_v<MakeSigned<uint16_t>::type, int16_t>);
static_assert(IsSame_v<MakeSigned<int32_t>::type,  int32_t>);
static_assert(IsSame_v<MakeSigned<uint32_t>::type, int32_t>);
static_assert(IsSame_v<MakeSigned<uint64_t>::type, int64_t>);
static_assert(IsSame_v<MakeSigned<uint64_t>::type, int64_t>);
static_assert(IsSame_v<MakeSigned_t<int8_t>,   int8_t>);
static_assert(IsSame_v<MakeSigned_t<uint8_t>,  int8_t>);
static_assert(IsSame_v<MakeSigned_t<int16_t>,  int16_t>);
static_assert(IsSame_v<MakeSigned_t<uint16_t>, int16_t>);
static_assert(IsSame_v<MakeSigned_t<int32_t>,  int32_t>);
static_assert(IsSame_v<MakeSigned_t<uint32_t>, int32_t>);
static_assert(IsSame_v<MakeSigned_t<int64_t>,  int64_t>);
static_assert(IsSame_v<MakeSigned_t<uint64_t>, int64_t>);

static_assert(IsSame_v<MakeUnsigned<int8_t>::type,   uint8_t>);
static_assert(IsSame_v<MakeUnsigned<uint8_t>::type,  uint8_t>);
static_assert(IsSame_v<MakeUnsigned<int16_t>::type,  uint16_t>);
static_assert(IsSame_v<MakeUnsigned<uint16_t>::type, uint16_t>);
static_assert(IsSame_v<MakeUnsigned<int32_t>::type,  uint32_t>);
static_assert(IsSame_v<MakeUnsigned<uint32_t>::type, uint32_t>);
static_assert(IsSame_v<MakeUnsigned<int64_t>::type,  uint64_t>);
static_assert(IsSame_v<MakeUnsigned<uint64_t>::type, uint64_t>);
static_assert(IsSame_v<MakeUnsigned_t<int8_t>,   uint8_t>);
static_assert(IsSame_v<MakeUnsigned_t<uint8_t>,  uint8_t>);
static_assert(IsSame_v<MakeUnsigned_t<int16_t>,  uint16_t>);
static_assert(IsSame_v<MakeUnsigned_t<uint16_t>, uint16_t>);
static_assert(IsSame_v<MakeUnsigned_t<int32_t>,  uint32_t>);
static_assert(IsSame_v<MakeUnsigned_t<uint32_t>, uint32_t>);
static_assert(IsSame_v<MakeUnsigned_t<int64_t>,  uint64_t>);
static_assert(IsSame_v<MakeUnsigned_t<uint64_t>, uint64_t>);

}  // namespace nvidia
