/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "common/expected.hpp"

#include <string>

#include "gtest/gtest.h"

namespace {
static uint32_t kConstruction;
static uint32_t kCopy;
static uint32_t kMove;
static uint32_t kDestruction;

void ResetCounters() {
  kConstruction = 0;
  kCopy = 0;
  kMove = 0;
  kDestruction = 0;
}

// Check static counters for construction operations
void TestCounters(uint32_t construction, uint32_t copy, uint32_t move, uint32_t destroy) {
    EXPECT_EQ(kConstruction, construction);
    EXPECT_EQ(kCopy, copy);
    EXPECT_EQ(kMove, move);
    EXPECT_EQ(kDestruction, destroy);
}

// Helper classes for testing construction and destruction behavior
struct Counter {
  Counter() = delete;
  Counter(const std::string& s, int d) : some(s), data(d) { ++kConstruction; }
  Counter(const Counter& o) : some(o.some), data(o.data) { ++kCopy; }
  Counter(Counter&& o) : some(std::move(o.some)), data(o.data)  { ++kMove; }
  ~Counter() { ++kDestruction; }
  Counter& operator=(const Counter& other) {
    ++kCopy;
    some = other.some;
    data = other.data;
    return *this;
  }
  Counter& operator=(Counter&& other) {
    ++kMove;
    some = std::move(other.some);
    data = other.data;
    return *this;
  }

  std::string some;
  int data;
};

struct NoCopy : public Counter {
  NoCopy() : Counter("", 0) {}
  NoCopy(const NoCopy&) = delete;
  NoCopy(NoCopy&&) = default;
  NoCopy& operator=(const NoCopy&) = delete;
  NoCopy& operator=(NoCopy&&) = default;
};

static_assert(!std::is_default_constructible<Counter>::value, "Counter default is constructible");
static_assert( std::is_default_constructible<NoCopy>::value, "NoCopy default is not constructible");

static_assert( std::is_copy_constructible<Counter>::value, "Counter copy is not constructible");
static_assert(!std::is_copy_constructible<NoCopy>::value, "NoCopy copy is constructible");

static_assert( std::is_copy_assignable<Counter>::value, "Counter copy is not assignable");
static_assert(!std::is_copy_assignable<NoCopy>::value, "NoCopy copy is assignable");

static_assert( std::is_move_constructible<Counter>::value, "Counter move is not constructible");
static_assert( std::is_move_constructible<NoCopy>::value, "NoCopy move is not constructible");

static_assert( std::is_move_assignable<Counter>::value, "Counter move is not assignable");
static_assert( std::is_move_assignable<NoCopy>::value, "NoCopy move is not assignable");
}

TEST(Expected, CopyConstructor_PrimitiveType) {
  nvidia::Expected<double, int> hello{3.14};

  nvidia::Expected<double, int> hello2{hello};
  ASSERT_TRUE(hello2);
  ASSERT_TRUE(hello2.has_value());
  ASSERT_NEAR(hello2.value(), 3.14, 1e-9);
}

TEST(Expected, ConstCopyConstructor_PrimitiveType) {
  const nvidia::Expected<double, int> hello{3.14};

  nvidia::Expected<double, int> hello2{hello};
  ASSERT_TRUE(hello2);
  ASSERT_TRUE(hello2.has_value());
  ASSERT_NEAR(hello2.value(), 3.14, 1e-9);
}

TEST(Expected, MoveAssignmentConstructor_PrimitiveType) {
  nvidia::Expected<double, int> hello{3.14};

  nvidia::Expected<double, int> hello2{std::move(hello)};
  ASSERT_TRUE(hello2);
  ASSERT_TRUE(hello2.has_value());
  ASSERT_NEAR(hello2.value(), 3.14, 1e-9);
}

TEST(Expected, ValueConstructor_PrimitiveType) {
  nvidia::Expected<double, int> hello{3.14};
  ASSERT_TRUE(hello);
  ASSERT_TRUE(hello.has_value());
  ASSERT_NEAR(hello.value(), 3.14, 1e-9);
}

TEST(Expected, UnexpectedConstructor_PrimitiveType) {
  nvidia::Expected<double, int> hello{nvidia::Unexpected<int>{3}};
  ASSERT_FALSE(hello);
  ASSERT_FALSE(hello.has_value());
  ASSERT_EQ(hello.error(), 3);
}

TEST(Expected, UnexpectedMoveConstructor_PrimitiveType) {
  nvidia::Unexpected<int> error{3};
  nvidia::Expected<double, int> hello{std::move(error)};
  ASSERT_FALSE(hello);
  ASSERT_FALSE(hello.has_value());
  ASSERT_EQ(hello.error(), 3);
}

TEST(Expected, ValueConstructor) {
  {
    ResetCounters();
    nvidia::Expected<Counter, std::string> hello{Counter{"hello", 42}};

    // one construct/destruct for the temporary, plus a move into the container
    TestCounters(1, 0, 1, 1);

    ASSERT_TRUE(hello);
    ASSERT_TRUE(hello.has_value());
    ASSERT_STREQ(hello.value().some.c_str(), "hello");
    ASSERT_EQ(hello.value().data, 42);
  }
  TestCounters(1, 0, 1, 2);
}

TEST(Expected, CopyConstructor) {
  ResetCounters();
  nvidia::Expected<Counter, std::string> hello{Counter{"hello", 42}};
  TestCounters(1, 0, 1, 1);

  nvidia::Expected<Counter, std::string> hello2{hello};
  TestCounters(1, 1, 1, 1);

  ASSERT_TRUE(hello2);
  ASSERT_TRUE(hello2.has_value());
  ASSERT_STREQ(hello2.value().some.c_str(), "hello");
  ASSERT_EQ(hello2.value().data, 42);
}

TEST(Expected, ConversionConstructors) {
  struct Convert {
    Convert() = delete;
    Convert(const Counter& o) : value(o.data) {}
    Convert(Counter&& o) : value(o.data) {}

    explicit Convert(const std::string& s) : value(s.size() + 1) {}
    explicit Convert(std::string&& s) : value(s.size() + 2) {}

    Convert(const std::string& s, int x) : value(s.size() + x) {}

    int value;
  };

  static_assert(!std::is_convertible<std::string, Convert>::value, "constructor is explicit");
  static_assert( std::is_constructible<Convert, std::string>::value, "constructor is valid");
  static_assert( std::is_constructible<Convert, std::string, int>::value, "constructor is valid");

  {  // Implicit conversion from different U
    ResetCounters();
    nvidia::Expected<Convert, std::string> fortyTwo{Counter{"throw away", 42}};
    TestCounters(1, 0, 0, 1);

    ASSERT_TRUE(fortyTwo);
    ASSERT_EQ(fortyTwo.value().value, 42);
  }

  {  // Implicit conversion from Expected<U>
    ResetCounters();
    nvidia::Expected<Counter, std::string> counter{"throw away", 42};
    TestCounters(1, 0, 0, 0);
    ASSERT_TRUE(counter);
    ASSERT_TRUE(counter.has_value());

    nvidia::Expected<Convert, std::string> fortyTwo{counter};
    TestCounters(1, 0, 0, 0);

    ASSERT_TRUE(fortyTwo);
    ASSERT_EQ(fortyTwo.value().value, 42);
  }

  {  // Implicit conversion from Expected<U>&&
    ResetCounters();
    nvidia::Expected<Counter, std::string> counter{"throw away", 42};
    TestCounters(1, 0, 0, 0);
    ASSERT_TRUE(counter);
    ASSERT_TRUE(counter.has_value());

    nvidia::Expected<Convert, std::string> fortyTwo{std::move(counter)};
    TestCounters(1, 0, 0, 0);

    ASSERT_TRUE(fortyTwo);
    ASSERT_EQ(fortyTwo.value().value, 42);
  }

  {  // Implicit conversion assignment from Expected<U>
    ResetCounters();
    nvidia::Expected<Counter, std::string> counter{"throw away", 42};
    TestCounters(1, 0, 0, 0);
    ASSERT_TRUE(counter);
    ASSERT_TRUE(counter.has_value());

    nvidia::Expected<Convert, std::string> fortyTwo = nvidia::Unexpected<std::string>{"not ready"};
    TestCounters(1, 0, 0, 0);

    fortyTwo = counter;
    TestCounters(1, 0, 0, 0);
    ASSERT_TRUE(fortyTwo);
    ASSERT_EQ(fortyTwo.value().value, 42);
  }

  {  // Implicit conversion assignment from Expected<U>&&
    ResetCounters();
    nvidia::Expected<Counter, std::string> counter{"throw away", 42};
    TestCounters(1, 0, 0, 0);
    ASSERT_TRUE(counter);
    ASSERT_TRUE(counter.has_value());

    nvidia::Expected<Convert, std::string> fortyTwo = nvidia::Unexpected<std::string>{"not ready"};
    TestCounters(1, 0, 0, 0);

    fortyTwo = std::move(counter);
    TestCounters(1, 0, 0, 0);
    ASSERT_TRUE(fortyTwo);
    ASSERT_EQ(fortyTwo.value().value, 42);
  }

  {  // Explicit conversion assignment from Expected<U>
    ResetCounters();
    nvidia::Expected<std::string, std::string> maybe_string = "";
    ASSERT_TRUE(maybe_string);

    nvidia::Expected<Convert, std::string> one{maybe_string};

    ASSERT_TRUE(one);
    ASSERT_EQ(one.value().value, 1);
  }

  {  // Explicit conversion assignment from Expected<U>&&
    ResetCounters();
    nvidia::Expected<std::string, std::string> maybe_string = "";;
    ASSERT_TRUE(maybe_string);

    nvidia::Expected<Convert, std::string> one{std::move(maybe_string)};

    ASSERT_TRUE(one);
    ASSERT_EQ(one.value().value, 2);
  }
}

TEST(Expected, ConstCopyConstructor) {
  ResetCounters();
  const nvidia::Expected<Counter, std::string> hello{Counter{"hello", 42}};
  TestCounters(1, 0, 1, 1);

  nvidia::Expected<Counter, std::string> hello2{hello};
  TestCounters(1, 1, 1, 1);

  ASSERT_TRUE(hello2);
  ASSERT_TRUE(hello2.has_value());
  ASSERT_STREQ(hello2.value().some.c_str(), "hello");
  ASSERT_EQ(hello2.value().data, 42);
}

TEST(Expected, MoveConstructor) {
  ResetCounters();
  nvidia::Expected<Counter, std::string> hello{Counter{"hello", 42}};
  TestCounters(1, 0, 1, 1);

  nvidia::Expected<Counter, std::string> hello2{std::move(hello)};
  TestCounters(1, 0, 2, 1);

  ASSERT_TRUE(hello2);
  ASSERT_TRUE(hello2.has_value());
  ASSERT_STREQ(hello2.value().some.c_str(), "hello");
  ASSERT_EQ(hello2.value().data, 42);

  // Implicit constructor of a move only object called from return
  auto get_moveable = [] () -> nvidia::Expected<NoCopy, int> {
    NoCopy retv{};
    return retv;
  };

  EXPECT_TRUE(get_moveable());
}

TEST(Expected, MoveAssignment) {
  Counter move_counter{"move", 55};
  nvidia::Expected<Counter, std::string> hello = nvidia::Unexpected<std::string>{""};

  ResetCounters();
  hello = Counter{"hello", 42};
  TestCounters(1, 0, 1, 1);
  ASSERT_TRUE(hello);
  ASSERT_EQ(hello.value().data, 42);
  ASSERT_STREQ(hello.value().some.c_str(), "hello");

  hello = std::move(move_counter);
  TestCounters(1, 0, 2, 2);

  ASSERT_TRUE(hello);
  ASSERT_EQ(hello.value().data, 55);
  ASSERT_STREQ(hello.value().some.c_str(), "move");
}

TEST(Expected, ReferenceConstructor) {
  Counter hello{"hello", 42};
  ResetCounters();

  nvidia::Expected<Counter&, std::string> hello_ref{hello};
  TestCounters(0, 0, 0, 0);

  ASSERT_TRUE(hello_ref);
  ASSERT_TRUE(hello_ref.has_value());
  ASSERT_STREQ(hello_ref.value().some.c_str(), "hello");
  ASSERT_EQ(hello_ref.value().data, 42);
  hello_ref.value().data++;
  hello.some = "world";
  ASSERT_EQ(hello.data, 43);
  ASSERT_STREQ(hello_ref.value().some.c_str(), "world");
}

TEST(Expected, ReferenceRebind) {
  Counter hello{"hello", 42};
  Counter earl{"Earl", -1};
  ResetCounters();

  nvidia::Expected<Counter&, std::string> ref{hello};
  TestCounters(0, 0, 0, 0);

  ASSERT_TRUE(ref);
  ASSERT_TRUE(ref.has_value());
  ASSERT_STREQ(ref.value().some.c_str(), "hello");
  ASSERT_EQ(ref.value().data, 42);

  ref = nvidia::Expected<Counter&, std::string>{earl};
  ref.value().data++;

  ASSERT_EQ(hello.data, 42);
  ASSERT_EQ(earl.data, 0);
  ASSERT_STREQ(ref.value().some.c_str(), "Earl");

  struct Hand {
   public:
    void open() { is_open_ = true; }
    void close() { is_open_ = false; }
    bool isOpen() const { return is_open_; }
   private:
    bool is_open_ = true;
  };

  Hand right_hand;
  Hand left_hand;
  nvidia::Expected<Hand&, std::string> my_hand = right_hand;
  EXPECT_TRUE(right_hand.isOpen());

  Hand& hand = *my_hand;
  hand.close();
  ASSERT_TRUE(my_hand);
  EXPECT_FALSE(right_hand.isOpen());
  EXPECT_FALSE(hand.isOpen());
  EXPECT_FALSE(my_hand->isOpen());

  my_hand = left_hand;
  EXPECT_FALSE(hand.isOpen());
  EXPECT_FALSE(right_hand.isOpen());
  EXPECT_TRUE(my_hand->isOpen());
}

TEST(Expected, ErrorTypeConversionConstructor) {
  // construction from const&
  nvidia::Expected<int, int8_t> maybe_char = nvidia::Unexpected<int8_t>(5);
  nvidia::Expected<int, int32_t> maybe_int(maybe_char);

  ASSERT_FALSE(maybe_int);
  EXPECT_EQ(maybe_int.error(), 5);

  // construction from prvalue
  nvidia::Expected<int, int32_t> maybe_int2{
      nvidia::Expected<int, int8_t>(nvidia::Unexpected<int8_t>(10))};

  ASSERT_FALSE(maybe_int2);
  EXPECT_EQ(maybe_int2.error(), 10);

  // assignment from const&
  maybe_char = nvidia::Unexpected<int8_t>(15);
  maybe_int = nvidia::Expected<int, int32_t>{maybe_char};

  ASSERT_FALSE(maybe_int);
  EXPECT_EQ(maybe_int.error(), 15);

  // assignment from prvalue
  maybe_int2 = nvidia::Expected<int, int32_t>(nvidia::Unexpected<int8_t>(20));

  ASSERT_FALSE(maybe_int2);
  EXPECT_EQ(maybe_int2.error(), 20);
}

TEST(Expected, Replace) {
  ResetCounters();
  nvidia::Expected<Counter, std::string> hello{Counter{"hello", 42}};
  TestCounters(1, 0, 1, 1);

  const Counter& goodbye = hello.replace("goodbye", 99);
  TestCounters(2, 0, 1, 2);

  ASSERT_TRUE(hello);
  ASSERT_STREQ(hello->some.c_str(), "goodbye");
  ASSERT_EQ(hello->data, 99);

  hello.replace("wait a sec", -5);
  ASSERT_EQ(goodbye.data, -5);
}

TEST(Expected, AddressLaunder) {
  struct A {
    const int a;
    int b;
  };

  A x{1,2};
  A y{3,4};

  nvidia::Expected<A, std::string> foo = nvidia::Unexpected<std::string>("default");
  foo = x;

  ASSERT_TRUE(foo);
  ASSERT_TRUE(foo.has_value());
  EXPECT_EQ(foo.value().a, 1);
  EXPECT_EQ(foo.value().b, 2);

  foo = y;
  ASSERT_TRUE(foo.has_value());
  EXPECT_EQ(foo.value().a, 3);
  EXPECT_EQ(foo.value().b, 4);
}

TEST(Expected, UnexpectedConstructor) {
  ResetCounters();
  nvidia::Expected<Counter, std::string> hello{nvidia::Unexpected<std::string>{"autsch"}};
  TestCounters(0, 0, 0, 0);

  ASSERT_FALSE(hello);
  ASSERT_FALSE(hello.has_value());
  ASSERT_STREQ(hello.error().c_str(), "autsch");
}

TEST(Expected, UnexpectedMoveConstructor) {
  ResetCounters();
  nvidia::Unexpected<std::string> error{"autsch"};
  nvidia::Expected<Counter, std::string> hello{std::move(error)};
  TestCounters(0, 0, 0, 0);

  ASSERT_FALSE(hello);
  ASSERT_FALSE(hello.has_value());
  ASSERT_STREQ(hello.error().c_str(), "autsch");
}

TEST(Expected, UnexpectedCopyAssignment) {
  NoCopy movable1, movable2;
  ResetCounters();
  TestCounters(0, 0, 0, 0);

  {
    nvidia::Expected<NoCopy, std::string> maybe{std::move(movable1)};
    TestCounters(0, 0, 1, 0);

    maybe = nvidia::Unexpected<std::string>{"replaced"};
    TestCounters(0, 0, 1, 1);

    maybe = std::move(movable2);
    TestCounters(0, 0, 2, 1);
  }

  TestCounters(0, 0, 2, 2);
}

TEST(Expected, RefCopyValueAssignment) {
  // Cannot construct for a prvalue
  static_assert(!nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>, NoCopy>);

  // Cannot construct mut& from const&
  static_assert(!nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>, const NoCopy&>);

  static_assert(!nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>,
                                           const nvidia::Expected<NoCopy, int>&>);

  static_assert(!nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>,
                                           nvidia::Expected<const NoCopy&, int>&>);

  // Can construct const& from mut&
  static_assert( nvidia::IsConstructible_v<nvidia::Expected<const NoCopy&, int>, NoCopy&>);

  static_assert( nvidia::IsConstructible_v<nvidia::Expected<const NoCopy&, int>,
                                           nvidia::Expected<NoCopy, int>&>);

  static_assert( nvidia::IsConstructible_v<nvidia::Expected<const NoCopy&, int>,
                                           nvidia::Expected<NoCopy&, int>&>);

  // Can construct from const Expected iff mut&
  static_assert( nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>,
                                           const nvidia::Expected<NoCopy&, int>&>);

  static_assert(!nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>,
                                           const nvidia::Expected<const NoCopy&, int>&>);

  // Can construct mut& from a prvalue Expected iff mut&
  static_assert( nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>,
                                           nvidia::Expected<NoCopy&, int>>);

  static_assert(!nvidia::IsConstructible_v<nvidia::Expected<NoCopy&, int>,
                                           nvidia::Expected<const NoCopy&, int>>);
  // Can construct const& from a prvalue Expected if also &
  static_assert( nvidia::IsConstructible_v<nvidia::Expected<const NoCopy&, int>,
                                           nvidia::Expected<const NoCopy&, int>>);

  static_assert( nvidia::IsConstructible_v<nvidia::Expected<const NoCopy&, int>,
                                           nvidia::Expected<NoCopy&, int>>);

  NoCopy refOnly;
  ResetCounters();
  TestCounters(0, 0, 0, 0);

  {
    nvidia::Expected<NoCopy&, std::string> maybe{refOnly};
    TestCounters(0, 0, 0, 0);

    maybe = nvidia::Unexpected<std::string>{"replaced"};
    TestCounters(0, 0, 0, 0);

    maybe = refOnly;
    TestCounters(0, 0, 0, 0);

    nvidia::Expected<NoCopy&, std::string> maybe2(nvidia::Unexpected<std::string>{"gone"});
    TestCounters(0, 0, 0, 0);
    EXPECT_FALSE(maybe2);
  }

  TestCounters(0, 0, 0, 0);
}

TEST(Expected, RefCopyExpectedAssignment) {
  // Runtime value checks
  NoCopy refOnly;
  const NoCopy crefOnly;

  NoCopy tmp;
  tmp.data = 55;
  const nvidia::Expected<NoCopy, int> const_maybe_nocopy(std::move(tmp));
  EXPECT_EQ(const_maybe_nocopy->data, 55);

  nvidia::Expected<NoCopy, int> maybe_nocopy(NoCopy{});
  ResetCounters();
  TestCounters(0, 0, 0, 0);

  {
    nvidia::Expected<NoCopy&, int> maybe{refOnly};
    TestCounters(0, 0, 0, 0);
    EXPECT_TRUE(maybe);

    maybe = nvidia::Unexpected<int>{234};
    TestCounters(0, 0, 0, 0);
    EXPECT_FALSE(maybe);

    maybe = nvidia::Expected<NoCopy&, int>{refOnly};
    TestCounters(0, 0, 0, 0);

    EXPECT_TRUE(maybe);

    maybe = maybe_nocopy;
    TestCounters(0, 0, 0, 0);
    EXPECT_TRUE(maybe);

    nvidia::Expected<const NoCopy&, int> maybe_const{refOnly};
    TestCounters(0, 0, 0, 0);
    EXPECT_TRUE(maybe_const);

    maybe_const = crefOnly;
    TestCounters(0, 0, 0, 0);
    EXPECT_TRUE(maybe_const);

    maybe_const = const_maybe_nocopy;
    TestCounters(0, 0, 0, 0);
    ASSERT_TRUE(maybe_const);
    EXPECT_EQ(maybe_const->data, 55);

    // Expected<const T&> from Expected<T&>
    maybe_const = maybe;
    TestCounters(0, 0, 0, 0);
    ASSERT_TRUE(maybe_const);
    EXPECT_EQ(maybe_const->data, 0);

    // Updating original mutates a const ref
    maybe->data = 22;
    EXPECT_EQ(maybe_const->data, 22);

  }

  TestCounters(0, 0, 0, 0);
}

TEST(Expected, ExpectedRefValue) {
  // Require conversion to Expected<void> to be explicit
  static_assert(nvidia::IsConstructible_v<nvidia::Expected<void, int>, nvidia::Expected<int, int>>);
  static_assert(!nvidia::IsConvertible_v<nvidia::Expected<int, int>, nvidia::Expected<void, int>>);

  int x = 5;

  // Test lvalue assignment
  nvidia::Expected<int&, std::string> maybe{x};
  ASSERT_TRUE(maybe);
  maybe.value() = 6;
  EXPECT_EQ(x, 6);

  auto wrap_ref = [](int& x) { return nvidia::Expected<int&, std::string>{x}; };

  // test rvalue assignment
  wrap_ref(x).value() = 8;
  EXPECT_EQ(x, 8);

  // construct from expected
  nvidia::Expected<int, std::string> other_int(1);
  nvidia::Expected<int&, std::string> ref(other_int);
  ASSERT_TRUE(ref);
  EXPECT_EQ(ref.value(), 1);
  ref.value()++;
  EXPECT_EQ(other_int.value(), 2);
  EXPECT_EQ(ref.value(), 2);
}

TEST(Expected, ExpectedVoid) {
  // Default construction
  nvidia::Expected<void, int> maybe0{};
  EXPECT_TRUE(maybe0);

  // lvalue construction from Expected<void>
  nvidia::Expected<void, int> maybe1{maybe0};
  EXPECT_TRUE(maybe1);

  // rvalue construction from Expected<void>
  nvidia::Expected<void, int> maybe2{std::move(maybe0)};
  EXPECT_TRUE(maybe2);

  // Explicit construction from not-void
  nvidia::Expected<void, int> maybe3{nvidia::Expected<int, int>{7}};
  EXPECT_TRUE(maybe3);
}

TEST(Expected, BitwiseAssignmentOperators) {
  nvidia::Expected<void, int> result;
  ASSERT_TRUE(result);

  result &= nvidia::Expected<void, int>{};
  ASSERT_TRUE(result);

  result &= nvidia::Expected<void, int>{nvidia::Unexpected<int>{3}};
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 3);

  result &= nvidia::Expected<void, int>{nvidia::Unexpected<int>{5}};
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 3);

  result |= nvidia::Expected<void, int>{};
  ASSERT_TRUE(result);

  result &= nvidia::Expected<void, int>{nvidia::Unexpected<int>{7}};
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 7);

  result |= nvidia::Expected<int, int>{0};
  ASSERT_TRUE(result);

  result &= nvidia::Expected<int, int>{2};
  ASSERT_TRUE(result);

  result &= nvidia::Expected<int, int>{nvidia::Unexpected<int>{4}};
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 4);

  result &= nvidia::Expected<int, int>{nvidia::Unexpected<int>{6}};
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 4);

  result = nvidia::Expected<void, int>{};
  result |= nvidia::Expected<int, int>{nvidia::Unexpected<int>{1}};
  ASSERT_TRUE(result);

  result = nvidia::Expected<void, int>{nvidia::Unexpected<int>{8}};
  result |= nvidia::Expected<int, int>{nvidia::Unexpected<int>{9}};
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 8);
}

TEST(Expected, BitOperatorsVoid) {
  nvidia::Expected<void, int> expected;
  const nvidia::Expected<void, int> unexpected1 = nvidia::Unexpected<int>{1};
  const nvidia::Expected<void, int> unexpected2 = nvidia::Unexpected<int>{2};
  const nvidia::Expected<void, int> unexpected3 = nvidia::Unexpected<int>{3};

  ASSERT_TRUE(expected & expected);
  ASSERT_FALSE(unexpected1 & unexpected2);
  ASSERT_FALSE(expected & unexpected2);
  ASSERT_FALSE(unexpected3 & expected);
  ASSERT_EQ((unexpected1 & unexpected2).error(), 1);
  ASSERT_EQ((expected & unexpected2).error(), 2);
  ASSERT_EQ((unexpected3 & expected).error(), 3);

  ASSERT_TRUE(expected | expected);
  ASSERT_FALSE(unexpected1 | unexpected2);
  ASSERT_TRUE(expected | unexpected2);
  ASSERT_TRUE(unexpected1 | expected);
  ASSERT_EQ((unexpected1 |  unexpected2).error(), 1);
}

TEST(Expected, BitOperatorsInt) {
  nvidia::Expected<int, int> expected1(5);
  nvidia::Expected<int, int> expected2(6);
  const nvidia::Expected<int, int> unexpected1 = nvidia::Unexpected<int>{1};
  const nvidia::Expected<int, int> unexpected2 = nvidia::Unexpected<int>{2};
  const nvidia::Expected<int, int> unexpected3 = nvidia::Unexpected<int>{3};

  ASSERT_TRUE(expected1 & expected2);
  ASSERT_FALSE(unexpected1 & unexpected2);
  ASSERT_FALSE(expected1 & unexpected2);
  ASSERT_FALSE(unexpected3 & expected1);
  ASSERT_EQ((unexpected1 & unexpected2).error(), 1);
  ASSERT_EQ((expected1 & unexpected2).error(), 2);
  ASSERT_EQ((unexpected3 & expected1).error(), 3);

  ASSERT_TRUE(expected1 | expected2);
  ASSERT_FALSE(unexpected1 | unexpected2);
  ASSERT_TRUE(expected1 | unexpected2);
  ASSERT_TRUE(unexpected1 | expected2);
  ASSERT_EQ((unexpected1 | unexpected2).error(), 1);
}

TEST(Expected, ExpectedArithmeticOperators) {
  nvidia::Expected<int, int> a{7};
  nvidia::Expected<int, int> b{3};
  ASSERT_EQ((a + b).value(), 10);
  ASSERT_EQ((a - b).value(), 4);
  ASSERT_EQ((a * b).value(), 21);
  ASSERT_EQ((a / b).value(), 2);
  ASSERT_EQ((a % b).value(), 1);

  nvidia::Expected<int, int> u{nvidia::Unexpected<int>{-1}};
  nvidia::Expected<int, int> v{nvidia::Unexpected<int>{-2}};
  ASSERT_EQ((u + a).error(), -1);
  ASSERT_EQ((u - a).error(), -1);
  ASSERT_EQ((u * a).error(), -1);
  ASSERT_EQ((u / a).error(), -1);
  ASSERT_EQ((u % a).error(), -1);
  ASSERT_EQ((a + v).error(), -2);
  ASSERT_EQ((a - v).error(), -2);
  ASSERT_EQ((a * v).error(), -2);
  ASSERT_EQ((a / v).error(), -2);
  ASSERT_EQ((a % v).error(), -2);
  ASSERT_EQ((u + v).error(), -1);
  ASSERT_EQ((u - v).error(), -1);
  ASSERT_EQ((u * v).error(), -1);
  ASSERT_EQ((u / v).error(), -1);
  ASSERT_EQ((u % v).error(), -1);
}

TEST(Expected, LeftExpectedArithmeticOperators) {
  nvidia::Expected<int, int> a{6};
  ASSERT_EQ((a + 3).value(), 9);
  ASSERT_EQ((a - 3).value(), 3);
  ASSERT_EQ((a * 3).value(), 18);
  ASSERT_EQ((a / 3).value(), 2);
  ASSERT_EQ((a % 5).value(), 1);

  nvidia::Expected<int, int> u{nvidia::Unexpected<int>{-1}};
  ASSERT_EQ((u + 3).error(), -1);
  ASSERT_EQ((u - 3).error(), -1);
  ASSERT_EQ((u * 3).error(), -1);
  ASSERT_EQ((u / 3).error(), -1);
  ASSERT_EQ((u % 3).error(), -1);
}

TEST(Expected, RightExpectedArithmeticOperators) {
  nvidia::Expected<int, int>  b{3};
  ASSERT_EQ((6 + b).value(), 9);
  ASSERT_EQ((6 - b).value(), 3);
  ASSERT_EQ((6 * b).value(), 18);
  ASSERT_EQ((6 / b).value(), 2);
  ASSERT_EQ((8 % b).value(), 2);

  nvidia::Expected<int, int> v{nvidia::Unexpected<int>{-2}};
  ASSERT_EQ((6 + v).error(), -2);
  ASSERT_EQ((6 - v).error(), -2);
  ASSERT_EQ((6 * v).error(), -2);
  ASSERT_EQ((6 / v).error(), -2);
  ASSERT_EQ((6 % v).error(), -2);
}

TEST(Expected, EqualsOperator) {
  const nvidia::Expected<int, int> value1{3};
  const nvidia::Expected<int, int> value2{3};
  const nvidia::Expected<int, int> value3{7};

  const nvidia::Unexpected<int> unexpected1{-2};
  const nvidia::Unexpected<int> unexpected2{-2};
  const nvidia::Unexpected<int> unexpected3{-9};

  const nvidia::Expected<int, int> error1 = unexpected1;
  const nvidia::Expected<int, int> error2 = unexpected2;
  const nvidia::Expected<int, int> error3 = unexpected3;

  EXPECT_TRUE(value1 == value2);
  EXPECT_FALSE(value1 == value3);

  EXPECT_TRUE(value1 == 3);
  EXPECT_TRUE(3 == value1);

  EXPECT_FALSE(value3 == 3);
  EXPECT_FALSE(3 == value3);

  EXPECT_FALSE(value1 == unexpected1);
  EXPECT_FALSE(unexpected2 == value2);

  EXPECT_TRUE(unexpected1 == unexpected2);
  EXPECT_FALSE(unexpected1 == unexpected3);

  EXPECT_FALSE(value1 == error1);
  EXPECT_FALSE(error2 == value2);

  EXPECT_TRUE(error1 == error2);
  EXPECT_FALSE(error1 == error3);
}

TEST(Expected, NotEqualsOperator) {
  const nvidia::Expected<int, int> value1{3};
  const nvidia::Expected<int, int> value2{3};
  const nvidia::Expected<int, int> value3{7};

  const nvidia::Unexpected<int> unexpected1{-2};
  const nvidia::Unexpected<int> unexpected2{-2};
  const nvidia::Unexpected<int> unexpected3{-9};

  const nvidia::Expected<int, int> error1 = unexpected1;
  const nvidia::Expected<int, int> error2 = unexpected2;
  const nvidia::Expected<int, int> error3 = unexpected3;

  EXPECT_FALSE(value1 != value2);
  EXPECT_TRUE(value1 != value3);

  EXPECT_FALSE(value1 != 3);
  EXPECT_FALSE(3 != value1);

  EXPECT_TRUE(value3 != 3);
  EXPECT_TRUE(3 != value3);

  EXPECT_TRUE(value1 != unexpected1);
  EXPECT_TRUE(unexpected2 != value2);

  EXPECT_FALSE(unexpected1 != unexpected2);
  EXPECT_TRUE(unexpected1 != unexpected3);

  EXPECT_TRUE(value1 != error1);
  EXPECT_TRUE(error2 != value2);

  EXPECT_FALSE(error1 != error2);
  EXPECT_TRUE(error1 != error3);
}

TEST(Expected, ConstReference) {
  Counter hello{"hello", 42};
  nvidia::Expected<const Counter&, std::string> const_ref{hello};

  ASSERT_TRUE(const_ref);
  ASSERT_TRUE(const_ref.has_value());
  ASSERT_STREQ(const_ref.value().some.c_str(), hello.some.c_str());
  ASSERT_EQ(const_ref.value().data, hello.data);
}

TEST(Expected, ValueOr) {
  Counter hello{"hello", 42};
  Counter world{"world", 43};

  nvidia::Expected<Counter&, std::string> ref{hello};
  nvidia::Expected<const Counter&, std::string> const_ref{hello};

  nvidia::Expected<Counter&, std::string> unexpected_ref{
    nvidia::Unexpected<std::string>{"olleh"}
  };
  nvidia::Expected<const Counter&, std::string> unexpected_const_ref{
    nvidia::Unexpected<std::string>{"olleh"}
  };

  Counter& result = ref.value_or(world);
  ASSERT_STREQ(result.some.c_str(), hello.some.c_str());
  ASSERT_EQ(result.data, hello.data);

  const Counter& const_result = const_ref.value_or(world);
  ASSERT_STREQ(const_result.some.c_str(), hello.some.c_str());
  ASSERT_EQ(const_result.data, hello.data);

  Counter& unexpected_result = unexpected_ref.value_or(world);
  ASSERT_STREQ(unexpected_result.some.c_str(), world.some.c_str());
  ASSERT_EQ(unexpected_result.data, world.data);

  const Counter& unexpected_const_result = unexpected_const_ref.value_or(world);
  ASSERT_STREQ(unexpected_const_result.some.c_str(), world.some.c_str());
  ASSERT_EQ(unexpected_const_result.data, world.data);
}

TEST(Expected, DereferenceArrow) {
  struct Hand {
   public:
    void open() { is_open_ = true; }
    void close() { is_open_ = false; }
    bool isOpen() const { return is_open_; }
   private:
    bool is_open_ = true;
  };

  nvidia::Expected<Hand, std::string> my_hand = Hand{};
  my_hand->close();
  ASSERT_FALSE(my_hand->isOpen());

  const nvidia::Expected<Hand&, std::string> left_hand = my_hand.value();
  my_hand->open();
  ASSERT_TRUE(my_hand->isOpen());

  Hand patricks_hand;
  nvidia::Expected<const Hand&, std::string> his_hand = patricks_hand;
  ASSERT_TRUE(his_hand->isOpen());

  nvidia::Expected<Hand, std::string> snake_hands =
      nvidia::Unexpected<std::string>{"snakes don't have hands"};

  ASSERT_DEATH(snake_hands->open(), "Expected does not have a value. Check before accessing.");
}

TEST(Expected, DereferenceStar) {
  struct Hand {
   public:
    void open() { is_open_ = true; }
    void close() { is_open_ = false; }
    bool isOpen() const { return is_open_; }
   private:
    bool is_open_ = true;
  };

  nvidia::Expected<Hand, std::string> my_hand = Hand{};

  Hand& hand = *my_hand;
  hand.close();
  ASSERT_FALSE(hand.isOpen());

  Hand patricks_hand;
  nvidia::Expected<const Hand&, std::string> his_hand = patricks_hand;

  const Hand& hand2 = *his_hand;
  ASSERT_TRUE(hand2.isOpen());

  nvidia::Expected<Hand, std::string> snake_hands =
      nvidia::Unexpected<std::string>{"snakes don't have hands"};

  ASSERT_DEATH(hand = *snake_hands, "Expected does not have a value. Check before accessing.");

  // Make sure we don't move a wrapped Lvalue when dereferencing a prvalue
  int ref = 5;
  auto always_value = [&ref]() { return nvidia::Expected<int&, int>{ref}; };
  int value = *always_value();
  int& ref2 = *always_value();
  ref = 10;
  ASSERT_EQ(value, 5);
  ASSERT_EQ(ref2, 10);
}

TEST(Expected, AndThen) {
  int n = 0;

  auto callback1 = [&]() { ++n; return nvidia::Expected<int, int>{5}; };
  auto callback2 = [&]() { ++n; return nvidia::Expected<void, int>{nvidia::Unexpected<int>{-1}}; };

  nvidia::Expected<int, int> result{1};
  ASSERT_TRUE(result);
  result = result.and_then(callback1);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 5);
  ASSERT_EQ(n, 1);
  const nvidia::Expected<void, int> result2 = result.and_then(callback2);
  ASSERT_FALSE(result2);
  ASSERT_EQ(result2.error(), -1);
  ASSERT_EQ(n, 2);
  result = result2.and_then(callback1);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), -1);
  ASSERT_EQ(n, 2);
}

TEST(Expected, MapNonary) {
  int n = 0;

  auto callback1 = [&]() { ++n; return nvidia::Expected<int, int>{5}; };
  auto callback2 = [&]() { ++n; return nvidia::Expected<void, int>{nvidia::Unexpected<int>{-1}}; };

  nvidia::Expected<int, int> result{1};
  ASSERT_TRUE(result);
  result = result.map(callback1);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 5);
  ASSERT_EQ(n, 1);
  const nvidia::Expected<void, int> result2 = result.map(callback2);
  ASSERT_FALSE(result2);
  ASSERT_EQ(result2.error(), -1);
  ASSERT_EQ(n, 2);
  result = result2.map(callback1);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), -1);
  ASSERT_EQ(n, 2);
}

TEST(Expected, AmbiguousMap) {
  struct Foo {
    int operator()() { return 5; }
    int operator()(int x) { return x; }
  };

  nvidia::Expected<int, int> result{1};
  ASSERT_TRUE(result);

  result = result.map(Foo{});
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 1);

  result = result.and_then(Foo{});
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 5);
}

TEST(Expected, Map) {
  int n = 0;
  nvidia::Unexpected<int> failure{-1};

  auto callback1 = [&](int x) { n+=x; return nvidia::Expected<int, int>{5}; };
  auto callback2 = [&](int) { ++n; return nvidia::Expected<void, int>{failure}; };
  auto callback3 = [&]() { n*=2; return nvidia::Expected<int, int>{failure}; };
  auto callback4 = [&](int& x) { n+=x; return nvidia::Expected<int, int>{++x}; };
  auto callback5 = [&](int&& x) { n = std::move(x); return nvidia::Expected<int, int>{55}; };

  nvidia::Expected<int, int> result{2};
  ASSERT_TRUE(result);
  result = result.map(callback1);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 5);
  ASSERT_EQ(n, 2);
  const nvidia::Expected<void, int> result2 = result.map(callback2);
  ASSERT_FALSE(result2);
  ASSERT_EQ(result2.error(), -1);
  ASSERT_EQ(n, 3);
  result = result2.map(callback3);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), -1);
  ASSERT_EQ(n, 3);
  result = 8;
  result = result.map(callback4);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 9);
  ASSERT_EQ(n, 11);
  result = result.map(callback1).map(callback5);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 55);
  ASSERT_EQ(n, 5);
}

TEST(Expected, MapRef) {
  int n = 0;
  int two = 2;
  int five = 5;
  int eight = 8;
  int fifty = 50;
  nvidia::Unexpected<int> failure{-1};

  auto callback1 = [&](int x) { n+=x; return nvidia::Expected<int&, int>{five}; };
  auto callback2 = [&](int) { ++n; return nvidia::Expected<void, int>{failure}; };
  auto callback3 = [&]() { n*=2; return nvidia::Expected<int&, int>{failure}; };
  auto callback4 = [&](int& x) { n+=x; return nvidia::Expected<int&, int>{++x}; };
  auto callback5 = [&](int&& x) { n = std::move(x); return nvidia::Expected<int&, int>{fifty}; };

  nvidia::Expected<int&, int> result{two};
  ASSERT_TRUE(result);
  result = result.map(callback1);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 5);
  ASSERT_EQ(n, 2);
  const nvidia::Expected<void, int> result2 = result.map(callback2);
  ASSERT_FALSE(result2);
  ASSERT_EQ(result2.error(), -1);
  ASSERT_EQ(n, 3);
  result = result2.map(callback3);
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), -1);
  ASSERT_EQ(n, 3);
  result = eight;
  result = result.map(callback4);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 9);
  ASSERT_EQ(n, 11);
  result = nvidia::Expected<int, int>{five}.map(callback5);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), 50);
  ASSERT_EQ(n, 5);
}

TEST(Expected, AllOf) {
  nvidia::Expected<void, int> e1, e2, e3;
  nvidia::Expected<int, int> u1 = nvidia::Unexpected<int>{-1};
  nvidia::Expected<int, int> u2 = nvidia::Unexpected<int>{-2};

  ASSERT_TRUE(AllOf(e1, e2, e3));
  ASSERT_FALSE(AllOf(e1, e2, u1));
  ASSERT_FALSE(AllOf(e1, u2, e1));
  ASSERT_FALSE(AllOf(u2, e2, e3));
  ASSERT_EQ(AllOf(u2, e2, u1).error(), -2);
  ASSERT_EQ(AllOf(u1, e2, u2).error(), -1);
}

TEST(Expected, Apply) {
  struct Record{ int n; };
  nvidia::Expected<int, int> e1{3};
  nvidia::Expected<std::string, int> e2{"Beetlejuice"};
  nvidia::Expected<Record, int> e3{Record{1}};

  nvidia::Expected<int, int> u1 = nvidia::Unexpected<int>{-1};
  nvidia::Expected<std::string, int> u2 = nvidia::Unexpected<int>{-2};

  auto f1 = [](int num, const std::string& str, Record& obj) {
    obj.n += num;
    std::string retv;
    for (int i = 0; i < num; ++i) retv += str;
    return  retv;
  };

  auto f2 = [](auto...) { ASSERT_FALSE(true); };

  nvidia::Expected<std::string, int> result = Apply(f1, e1, e2, e3);
  ASSERT_TRUE(result);
  ASSERT_EQ(result.value(), "BeetlejuiceBeetlejuiceBeetlejuice");
  ASSERT_EQ(e3.value().n, 4);

  nvidia::Expected<void, int> result2 = Apply(f2, u1, e2, e3);
  ASSERT_FALSE(result2);
  ASSERT_EQ(result2.error(), -1);

  nvidia::Expected<void, int> result3 = Apply(f2, e1, u2, e3);
  ASSERT_FALSE(result3);
  ASSERT_EQ(result3.error(), -2);
}

TEST(Expected, AssignTo) {
  ResetCounters();

  auto MaybeCounter = [](int x) -> nvidia::Expected<Counter, std::string> {
    if (x > 0) {
      // NOTE: this constructor creates a temporary, moves it into the expected, and then deletes
      // the temporary, so we need to account for added {1, 0, 1, 1} in TestCounters.
      return Counter("", x);
    }
    return nvidia::Unexpected<std::string>{"Spanish Inquisition"};
  };

  Counter my_counter("", 0);
  TestCounters(1, 0, 0, 0);

  // Dont Move Unexpected
  nvidia::Expected<void, std::string> result = MaybeCounter(0).assign_to(my_counter);
  EXPECT_FALSE(result);
  EXPECT_EQ(my_counter.data, 0);
  EXPECT_EQ(result.error(), "Spanish Inquisition");
  TestCounters(1, 0, 0, 0);

  // Move Expected
  result = MaybeCounter(2).assign_to(my_counter);
  EXPECT_TRUE(result);
  EXPECT_EQ(my_counter.data, 2);
  // NOTE: inlining MaybeCounter uses an extra move constructor/destructor call for the temporary
  // Expected value
  TestCounters(2, 0, 2, 2);

  // Don't Copy Unexpected
  const auto failure = MaybeCounter(0);
  result = failure.assign_to(my_counter);
  TestCounters(2, 0, 2, 2);
  EXPECT_FALSE(result);
  EXPECT_EQ(my_counter.data, 2);
  EXPECT_EQ(result.error(), "Spanish Inquisition");

  // Copy Expected
  const auto success = MaybeCounter(5);
  TestCounters(3, 0, 3, 3);
  result = success.assign_to(my_counter);
  EXPECT_TRUE(result);
  EXPECT_EQ(my_counter.data, 5);
  TestCounters(3, 1, 3, 3);
}

TEST(Expected, AssignToConversion) {
  struct Wrapper {
    Wrapper() = default;
    Wrapper(int x) : value(x) {}
    int value = -1;
  };

  auto MaybeInt = [](int x) -> nvidia::Expected<int, std::string> {
    if (x > 0) { return x; }
    return nvidia::Unexpected<std::string>{"Unnatural Number"};
  };

  Wrapper wrapper;

  // Rvalue assign_to
  auto result = MaybeInt(0).assign_to(wrapper);
  EXPECT_FALSE(result);
  EXPECT_EQ(wrapper.value, -1);

  result = MaybeInt(5).assign_to(wrapper);
  EXPECT_TRUE(result);
  EXPECT_EQ(wrapper.value, 5);

  // Lvalue assign_to
  const auto maybe0 = MaybeInt(0);
  result = maybe0.assign_to(wrapper);
  EXPECT_FALSE(result);
  EXPECT_EQ(wrapper.value, 5);

  const auto maybe9 = MaybeInt(9);
  result = maybe9.assign_to(wrapper);
  EXPECT_TRUE(result);
  EXPECT_EQ(wrapper.value, 9);
}

TEST(Expected, AssignToRefConversion) {
  struct Wrapper {
    Wrapper() = default;
    Wrapper(int x) : value(x) {}
    int value = -1;
  };

  auto MaybeInt = [](int& x) -> nvidia::Expected<int&, std::string> {
    if (x > 0) { return x; }
    return nvidia::Unexpected<std::string>{"Unnatural Number"};
  };

  int iref = 0;
  Wrapper nat;

  // Rvalue assign_to
  auto result = MaybeInt(iref).assign_to(nat);
  iref = 5;
  EXPECT_FALSE(result);
  EXPECT_EQ(nat.value, -1);

  result = MaybeInt(iref).assign_to(nat);
  iref = 0;
  EXPECT_TRUE(result);
  EXPECT_EQ(nat.value, 5);

  // Lvalue assign_to
  const auto maybe0 = MaybeInt(iref);
  iref = 9;
  result = maybe0.assign_to(nat);
  EXPECT_FALSE(result);
  EXPECT_EQ(nat.value, 5);

  const auto maybe9 = MaybeInt(iref);
  iref = 11;
  result = maybe9.assign_to(nat);
  EXPECT_TRUE(result);
  EXPECT_EQ(nat.value, 11);
}

TEST(Expected, AssignToRef) {
  ResetCounters();
  Counter my_counter_ref("", -1);

  auto MaybeCounter = [&](int x) -> nvidia::Expected<Counter&, std::string> {
    if (x > 0) {
      my_counter_ref.data = x;
      return my_counter_ref;
    }
    return nvidia::Unexpected<std::string>{"Spanish Inquisition"};
  };

  Counter my_counter("", 0);
  TestCounters(2, 0, 0, 0);

  // Rvalue Don't Copy Unexpected
  nvidia::Expected<void, std::string> result = MaybeCounter(0).assign_to(my_counter);
  EXPECT_FALSE(result);
  EXPECT_EQ(my_counter_ref.data, -1);
  EXPECT_EQ(my_counter.data, 0);
  EXPECT_EQ(result.error(), "Spanish Inquisition");
  TestCounters(2, 0, 0, 0);

  // Rvalue Move to Expected
  result = MaybeCounter(2).assign_to(my_counter);
  EXPECT_TRUE(result);
  EXPECT_EQ(my_counter_ref.data, 2);
  EXPECT_EQ(my_counter.data, 2);
  TestCounters(2, 0, 1, 0);

  // Lvalue Don't Copy Unexpected
  const auto failure = MaybeCounter(0);
  result = failure.assign_to(my_counter);
  TestCounters(2, 0, 1, 0);
  EXPECT_FALSE(result);
  EXPECT_EQ(my_counter_ref.data, 2);
  EXPECT_EQ(my_counter.data, 2);
  EXPECT_EQ(result.error(), "Spanish Inquisition");

  // Lvalue Copy Expected
  const auto success = MaybeCounter(5);
  TestCounters(2, 0, 1, 0);
  result = success.assign_to(my_counter);
  EXPECT_TRUE(result);
  EXPECT_EQ(my_counter_ref.data, 5);
  EXPECT_EQ(my_counter.data, 5);
  TestCounters(2, 1, 1, 0);
}

TEST(Expected, Substitute) {
  int value = 8;
  const int cvalue = 50;

  // Unexpected
  nvidia::Expected<void, int> maybe = nvidia::Unexpected<int>(0);
  nvidia::Expected<int, int> maybe_int = maybe.substitute(value);
  EXPECT_FALSE(maybe_int);

  // replace with value
  maybe = nvidia::Expected<void, int>();
  maybe_int = maybe.substitute(5);

  EXPECT_TRUE(maybe_int);
  EXPECT_EQ(maybe_int.value(), 5);

  // replace with copy
  maybe = nvidia::Expected<void, int>();
  maybe_int = maybe.substitute(value);
  value = 0;

  EXPECT_TRUE(maybe_int);
  EXPECT_EQ(maybe_int.value(), 8);

  // replace with copy from const value
  static_assert(std::is_same<decltype(maybe.substitute(cvalue)), nvidia::Expected<int, int>>::value,
      "substitute of const ref results in a copy by default");

  maybe = nvidia::Expected<void, int>();
  maybe_int = maybe.substitute(cvalue);

  EXPECT_TRUE(maybe_int);
  EXPECT_EQ(maybe_int.value(), 50);

  // replace as mutable ref
  nvidia::Expected<int&, int> maybe_ref = maybe.substitute<int&>(value);

  EXPECT_TRUE(maybe_ref);
  EXPECT_EQ(maybe_ref.value(), 0);

  maybe_ref.value() = 1;
  EXPECT_EQ(maybe_ref.value(), 1);
  EXPECT_EQ(value, 1);

  // replace as const ref
  nvidia::Expected<const int&, int> maybe_cref = maybe.substitute<const int&>(cvalue);

  EXPECT_TRUE(maybe_cref);
  EXPECT_EQ(maybe_cref.value(), 50);

  // replace with move
  NoCopy only_move;
  nvidia::Expected<NoCopy, int> maybe_move = maybe.substitute(std::move(only_move));

  EXPECT_TRUE(maybe_move);

  // Do not copy prvalue types
  maybe_move = maybe.substitute(NoCopy{});
  EXPECT_TRUE(maybe_move);

  // Flatten expected types
  nvidia::Expected<int, int> other_int(5);
  nvidia::Expected<int, int> flat = maybe_int.substitute(other_int);
  EXPECT_TRUE(flat);
  EXPECT_EQ(flat.value(), 5);

  other_int = nvidia::Unexpected<int>(909);
  flat = maybe_int.substitute(other_int);
  EXPECT_FALSE(other_int);
  EXPECT_FALSE(flat);
  EXPECT_EQ(flat.error(), 909);

  // Preserve first error code
  other_int = nvidia::Unexpected<int>(808);
  flat = flat.substitute(other_int);
  EXPECT_FALSE(flat);
  EXPECT_EQ(flat.error(), 909);

  // call from const&&
  const nvidia::Expected<int, int> const_int(5);
  nvidia::Expected<std::string, int> moved_string = std::move(const_int).substitute("moved");
  EXPECT_TRUE(moved_string);
  EXPECT_EQ(*moved_string, "moved");
}

TEST(Expected, SubstituteError) {
  auto even = [](int x) -> nvidia::Expected<int, int> {
    return (x & 0x1) == 0 ? nvidia::Expected<int, int>(x) : nvidia::Unexpected<int>(1);
  };

  auto positive = [](int x) -> nvidia::Expected<void, int> {
    return x > 0 ? nvidia::Expected<void, int>() : nvidia::Unexpected<int>(2);
  };

  // -- From Value types --
  // Expected const&
  nvidia::Expected<int, int> maybe = 10;
  nvidia::Expected<int, std::string> maybe_string = maybe.substitute_error<std::string>("fail");
  EXPECT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), 10);

  // Expected &&
  maybe_string = even(20).substitute_error<std::string>("fail");
  EXPECT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), 20);

  // Expected<void> const&
  nvidia::Expected<void, int> maybe_void;
  nvidia::Expected<void, std::string> maybe_void_string =
      maybe_void.substitute_error<std::string>("fail");
  EXPECT_TRUE(maybe_void_string);

  // Expected<void> &&
  maybe_void_string = positive(20).substitute_error<std::string>("not positive");
  EXPECT_TRUE(maybe_void_string);

  // -- From Unexpected types --
  // Expected const&
  maybe = nvidia::Unexpected<int>(0);
  maybe_string = maybe.substitute_error<std::string>("fail");
  EXPECT_FALSE(maybe_string);
  EXPECT_EQ(maybe_string.error(), "fail");

  // Expected &&
  maybe_string = even(31).substitute_error<std::string>("odd");
  EXPECT_FALSE(maybe_string);
  EXPECT_EQ(maybe_string.error(),"odd");

  // Expected<void> const&
  maybe_void = nvidia::Unexpected<int>(0);
  maybe_void_string = maybe_void.substitute_error<std::string>("I am the abyss");
  EXPECT_FALSE(maybe_void_string);
  EXPECT_EQ(maybe_void_string.error(), "I am the abyss");

  // Expected<void> &&
  maybe_void_string = positive(-20).substitute_error<std::string>("darkness consumes all");
  EXPECT_FALSE(maybe_void_string);
  EXPECT_EQ(maybe_void_string.error(), "darkness consumes all");
}

TEST(Expected, MapError) {
  auto even = [](int x) -> nvidia::Expected<int, int> {
    return (x & 0x1) == 0 ? nvidia::Expected<int, int>(x) : nvidia::Unexpected<int>(1);
  };

  auto positive = [](int x) -> nvidia::Expected<void, int> {
    return x > 0 ? nvidia::Expected<void, int>() : nvidia::Unexpected<int>(20);
  };

  auto error_str = [] (int error) -> std::string {
      return error < 5 ? "little error" : "big error";
  };

  // -- From Value types --
  // Expected &
  nvidia::Expected<int, int> maybe = 10;
  nvidia::Expected<int, std::string> maybe_string = maybe.map_error(error_str);

  EXPECT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), 10);

  // Expected &&
  maybe_string = even(20).map_error(error_str);
  EXPECT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), 20);

  // Expected const &
  const nvidia::Expected<int, int> const_maybe = 10;
  maybe_string = const_maybe.map_error(error_str);

  EXPECT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), 10);

  // Expected const &&
  const nvidia::Expected<int, int> maybe_even = even(20);
  maybe_string = std::move(maybe_even).map_error(error_str);
  EXPECT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), 20);

  // Expected<void> &
  nvidia::Expected<void, int> maybe_void;
  nvidia::Expected<void, std::string> maybe_void_string = maybe_void.map_error(error_str);
  EXPECT_TRUE(maybe_void_string);

  // Expected<void> &&
  maybe_void_string = positive(20).map_error(error_str);
  EXPECT_TRUE(maybe_void_string);

  // -- From Unexpected types --
  // Expected &
  maybe = nvidia::Unexpected<int>(0);
  maybe_string = maybe.map_error(error_str);
  EXPECT_FALSE(maybe_string);
  EXPECT_EQ(maybe_string.error(), "little error");

  // Expected &&
  maybe_string = even(31).map_error(error_str);
  EXPECT_FALSE(maybe_string);
  EXPECT_EQ(maybe_string.error(),"little error");

  // Expected const &
  const nvidia::Expected<int, int> const_error = nvidia::Unexpected<int>(-5);
  maybe_string = const_error.map_error(error_str);

  EXPECT_FALSE(maybe_string);
  EXPECT_EQ(maybe_string.error(),"little error");

  // Expected const &&
  const nvidia::Expected<int, int> maybe_not_even = even(41);
  maybe_string = std::move(maybe_not_even).map_error(error_str);
  EXPECT_FALSE(maybe_string);
  EXPECT_EQ(maybe_string.error(),"little error");

  // Expected<void> &
  maybe_void = nvidia::Unexpected<int>(40);
  maybe_void_string = maybe_void.map_error(error_str);
  EXPECT_FALSE(maybe_void_string);
  EXPECT_EQ(maybe_void_string.error(), "big error");

  // Expected<void> &&
  maybe_void_string = positive(-20).map_error(error_str);
  EXPECT_FALSE(maybe_void_string);
  EXPECT_EQ(maybe_void_string.error(), "big error");
}

TEST(Expected, AndThenError) {
  int n = 0;

  auto callback1 = [&]() { ++n; return 20; };
  auto callback2 = [&]() { ++n; return nvidia::Unexpected<int>{50}; };

  nvidia::Expected<int, int> result{1};
  ASSERT_TRUE(result);
  result = result.and_then_error(callback1);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value(), 1);
  EXPECT_EQ(n, 0);
  result = nvidia::Unexpected<int>{5};
  result = result.and_then_error(callback1);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error(), 20);
  EXPECT_EQ(n, 1);

  result = result.and_then_error(callback2);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error(), 50);
  EXPECT_EQ(n, 2);
}

TEST(Expected, MapNonaryError) {
  int n = 0;

  auto callback1 = [&]() { ++n; return 20; };
  auto callback2 = [&]() { ++n; return nvidia::Unexpected<int>{50}; };

  nvidia::Expected<int, int> result{1};
  ASSERT_TRUE(result);
  result = result.map_error(callback1);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.value(), 1);
  EXPECT_EQ(n, 0);
  result = nvidia::Unexpected<int>{5};
  result = result.map_error(callback1);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error(), 20);
  EXPECT_EQ(n, 1);

  result = result.map_error(callback2);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error(), 50);
  EXPECT_EQ(n, 2);
}

TEST(Expected, AmbiguousMapError) {
  struct Foo {
    int operator()() { return 5; }
    int operator()(int x) { return x; }
  };

  nvidia::Expected<int, int> result = nvidia::Unexpected<int>{1};
  ASSERT_FALSE(result);

  result = result.map_error(Foo{});
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 1);

  result = result.and_then_error(Foo{});
  ASSERT_FALSE(result);
  ASSERT_EQ(result.error(), 5);
}

TEST(Expected, LogError) {
  int log_count = 0;

  // Call from mutable ref. Update error to enforce mutable ref overload resolution.
  std::string error_str = "Failure";

  nvidia::Expected<void, std::string> maybe = nvidia::Unexpected<std::string>("Failure");
  maybe.log_error("create log item %d", log_count++).error() = "Replace";
  EXPECT_FALSE(maybe.has_value());
  EXPECT_EQ(maybe.error(), "Replace");

  // Call from const ref
  const nvidia::Expected<int, int> const_maybe = nvidia::Unexpected<int>(0);
  const nvidia::Expected<int, int> new_maybe =
      const_maybe.log_error("create log item %d", log_count++);

  // Call from rvalue ref
  const nvidia::Expected<NoCopy, int> build_maybe =
      nvidia::Expected<NoCopy, int>{nvidia::Unexpected<int>(0)}
      .log_error("create log item %d", log_count++);

  // NOTE: log count is always evaluated as an argument, regardless of the state of the Expected.
  EXPECT_EQ(log_count, 3);
}

TEST(Expected, LogWarning) {
  int log_count = 0;

  // Call from mutable ref. Update error to enforce mutable ref overload resolution.
  std::string error_str = "Failure";

  nvidia::Expected<void, std::string> maybe = nvidia::Unexpected<std::string>("Failure");
  maybe.log_warning("create log item %d", log_count++).error() = "Replace";
  EXPECT_FALSE(maybe.has_value());
  EXPECT_EQ(maybe.error(), "Replace");

  // Call from const ref
  const nvidia::Expected<int, int> const_maybe = nvidia::Unexpected<int>(0);
  const nvidia::Expected<int, int> new_maybe =
      const_maybe.log_warning("create log item %d", log_count++);

  // Call from rvalue ref
  const nvidia::Expected<NoCopy, int> build_maybe =
      nvidia::Expected<NoCopy, int>{nvidia::Unexpected<int>(0)}
      .log_warning("create log item %d", log_count++);

  // NOTE: log count is always evaluated as an argument, regardless of the state of the Expected.
  EXPECT_EQ(log_count, 3);
}

TEST(Expected, LogInfo) {
  int log_count = 0;

  // Call from mutable ref. Update error to enforce mutable ref overload resolution.
  std::string error_str = "Failure";

  nvidia::Expected<void, std::string> maybe = nvidia::Unexpected<std::string>("Failure");
  maybe.log_info("create log item %d", log_count++).error() = "Replace";
  EXPECT_FALSE(maybe.has_value());
  EXPECT_EQ(maybe.error(), "Replace");

  // Call from const ref
  const nvidia::Expected<int, int> const_maybe = nvidia::Unexpected<int>(0);
  const nvidia::Expected<int, int> new_maybe =
      const_maybe.log_info("create log item %d", log_count++);

  // Call from rvalue ref
  const nvidia::Expected<NoCopy, int> build_maybe =
      nvidia::Expected<NoCopy, int>{nvidia::Unexpected<int>(0)}
      .log_info("create log item %d", log_count++);

  // NOTE: log count is always evaluated as an argument, regardless of the state of the Expected.
  EXPECT_EQ(log_count, 3);
}

TEST(Expected, LogDebug) {
  int log_count = 0;

  // Call from mutable ref. Update error to enforce mutable ref overload resolution.
  std::string error_str = "Failure";

  nvidia::Expected<void, std::string> maybe = nvidia::Unexpected<std::string>("Failure");
  maybe.log_debug("create log item %d", log_count++).error() = "Replace";
  EXPECT_FALSE(maybe.has_value());
  EXPECT_EQ(maybe.error(), "Replace");

  // Call from const ref
  const nvidia::Expected<int, int> const_maybe = nvidia::Unexpected<int>(0);
  const nvidia::Expected<int, int> new_maybe =
      const_maybe.log_debug("create log item %d", log_count++);

  // Call from rvalue ref
  const nvidia::Expected<NoCopy, int> build_maybe =
      nvidia::Expected<NoCopy, int>{nvidia::Unexpected<int>(0)}
      .log_debug("create log item %d", log_count++);

  // NOTE: log count is always evaluated as an argument, regardless of the state of the Expected.
  EXPECT_EQ(log_count, 3);
}

TEST(Expected, MapMemberFunction) {
  struct A : NoCopy {
    int foo()                         { return 5; }
    int foo(int x) const              { return x; }
    int baz() &                       { return 10; }
    int baz() &&                      { return 11; }
    int baz() const&                  { return 12; }
    int baz() const&&                 { return 13; }
    std::string bar()                 { return "mutable"; }
    std::string bar() const           { return "const"; }
    nvidia::Expected<void, int> zap() { return {}; }
  };

  nvidia::Expected<A, int> maybe_a = A{};
  nvidia::Expected<int, int> maybe_int = maybe_a.map(&A::foo);
  EXPECT_TRUE(maybe_int);
  EXPECT_EQ(maybe_int.value(), 5);

  // With an argument list
  maybe_int = maybe_a.map(&A::foo, 88);
  EXPECT_TRUE(maybe_int);
  EXPECT_EQ(maybe_int.value(), 88);

  // Resolve ambiguous pointer declaration since using the expected const specifier. If the
  // expected is not const, this will fail.
  nvidia::Expected<std::string, int> maybe_string = maybe_a.map(&A::bar);
  ASSERT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), "mutable");

  const nvidia::Expected<A, int> const_maybe = A{};
  maybe_string = const_maybe.map(&A::bar);
  ASSERT_TRUE(maybe_string);
  EXPECT_EQ(maybe_string.value(), "const");

  // From lvalue
  auto maybe_baz = maybe_a.map(&A::baz);
  ASSERT_TRUE(maybe_baz);
  EXPECT_EQ(*maybe_baz, 10);

  // From rvalue
  maybe_baz = std::move(maybe_a).map(&A::baz);
  ASSERT_TRUE(maybe_baz);
  EXPECT_EQ(*maybe_baz, 11);

  // From const rvalue
  maybe_baz = const_maybe.map(&A::baz);
  ASSERT_TRUE(maybe_baz);
  EXPECT_EQ(*maybe_baz, 12);

  // From const rvalue
  maybe_baz = std::move(const_maybe).map(&A::baz);
  ASSERT_TRUE(maybe_baz);
  EXPECT_EQ(*maybe_baz, 13);

  nvidia::Expected<void, int> maybe_zap = maybe_a.map(&A::zap);
  ASSERT_TRUE(maybe_zap);
}

TEST(Expected, MapMemberVariable) {
  struct A {
    int bar = 10;
    const int cbar = 15;
    nvidia::Expected<int, int> baz = 25;
  };

  nvidia::Expected<A, int> maybe_a = A{.bar = -5};
  nvidia::Expected<int, int> maybe_int = maybe_a.map(&A::bar);
  // from T&
  auto maybe_int2 = maybe_a.map(&A::bar);
  static_assert(nvidia::IsSame_v<decltype(maybe_int2), nvidia::Expected<int&, int>>);

  EXPECT_TRUE(maybe_int2);
  EXPECT_EQ(maybe_int2.value(), -5);

  // Test that member variables are captured as reference by default.
  *maybe_int2 = -99;
  EXPECT_EQ(maybe_a->bar, -99);

  // from T&&
  // Member from a prvalue must return in a copy
  auto maybe_int3 = nvidia::Expected<A, int>(A{}).map(&A::cbar);
  static_assert(nvidia::IsSame_v<decltype(maybe_int3), nvidia::Expected<int, int>>);

  EXPECT_TRUE(maybe_int3);
  EXPECT_EQ(maybe_int3.value(), 15);

  // from const T&
  const nvidia::Expected<A, int> maybe_b = A{.bar = -77};
  auto maybe_int4 = maybe_b.map(&A::bar);
  static_assert(nvidia::IsSame_v<decltype(maybe_int4), nvidia::Expected<const int&, int>>);

  EXPECT_TRUE(maybe_int4);
  EXPECT_EQ(maybe_int4.value(), -77);

  // from const T&&
  // Member from a prvalue must return in a copy
  const nvidia::Expected<A, int> maybe_move{A{}};
  auto maybe_int5 = std::move(maybe_move).map(&A::cbar);
  static_assert(nvidia::IsSame_v<decltype(maybe_int5), nvidia::Expected<int, int>>);

  EXPECT_TRUE(maybe_int5);
  EXPECT_EQ(maybe_int5.value(), 15);

  nvidia::Expected<int&, int> maybe_baz = maybe_a.map(&A::baz);
  ASSERT_TRUE(maybe_baz);
  EXPECT_EQ(maybe_baz.value(), 25);
}

TEST(Expected, IgnoreErrorUnexpected) {
  auto even = [](int x) -> nvidia::Expected<int, int> {
    return (x & 0x1) == 0 ? nvidia::Expected<int, int>(x) : nvidia::Unexpected<int>(1);
  };

  // no arguments
  nvidia::Expected<int, int> maybe = nvidia::Unexpected<int>(5);
  EXPECT_FALSE(maybe);
  EXPECT_TRUE(maybe.ignore_error());

  // const lvalue
  auto maybe_value = maybe.ignore_error(111);
  EXPECT_TRUE(maybe_value);
  EXPECT_EQ(*maybe_value, 111);

  // rvalue
  maybe_value = even(5).ignore_error(22);
  EXPECT_TRUE(maybe_value);
  EXPECT_EQ(*maybe_value, 22);

  // const rvalue
  const nvidia::Expected<int, int> maybe_const = nvidia::Unexpected<int>(1);
  maybe_value = std::move(maybe_const).ignore_error(90);
  EXPECT_TRUE(maybe_value);
  EXPECT_EQ(*maybe_value, 90);
}

TEST(Expected, IgnoreErrorValue) {
  auto even = [](int x) {
    return x & 0x1 ? nvidia::Unexpected<int>(1) : nvidia::Expected<int, int>(x);
  };

  // no arguments
  nvidia::Expected<int, int> maybe = 5;
  EXPECT_TRUE(maybe);
  EXPECT_TRUE(maybe.ignore_error());

  // const lvalue
  auto maybe_value = maybe.ignore_error(111);
  EXPECT_TRUE(maybe_value);
  EXPECT_EQ(*maybe_value, 5);

  // rvalue
  maybe_value = even(40).ignore_error(22);
  EXPECT_TRUE(maybe_value);
  EXPECT_EQ(*maybe_value, 40);

  // const rvalue
  const nvidia::Expected<int, int> maybe_const = 90;
  maybe_value = std::move(maybe_const).ignore_error(22);
  EXPECT_TRUE(maybe_value);
  EXPECT_EQ(*maybe_value, 90);
}

TEST(Expected, Guard) {
  nvidia::Expected<int, int> maybe_int = 2;
  auto is_small = [](int x) { return x < 5; };
  // Pass Guard
  // const lvalue
  nvidia::Expected<int, int> maybe_small = maybe_int.guard(is_small, 99);
  EXPECT_TRUE(maybe_small);
  EXPECT_EQ(*maybe_small, 2);

  // rvalue
  maybe_small = nvidia::Expected<int, int>(-10).guard(is_small, 98);
  EXPECT_TRUE(maybe_small);
  EXPECT_EQ(*maybe_small, -10);

  // const rvalue
  const nvidia::Expected<int, int> maybe_const = 1;
  maybe_small = std::move(maybe_const).guard(is_small, 98);
  EXPECT_TRUE(maybe_small);
  EXPECT_EQ(*maybe_small, 1);

  // Fail Guard
  // const lvalue
  maybe_int = 10;
  maybe_small = maybe_int.guard(is_small, 97);
  EXPECT_FALSE(maybe_small);
  EXPECT_EQ(maybe_small.error(), 97);

  // rvalue
  maybe_small = nvidia::Expected<int, int>(50).guard(is_small, 96);
  EXPECT_FALSE(maybe_small);
  EXPECT_EQ(maybe_small.error(), 96);

  // const rvalue
  const nvidia::Expected<int, int> maybe_large_const = 100;
  maybe_small = std::move(maybe_large_const).guard(is_small, 55);
  EXPECT_FALSE(maybe_small);
  EXPECT_EQ(maybe_small.error(), 55);
}
