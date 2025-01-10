Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

# QueueThread
It is aimed to provide a unified Async Runnable Processing Thread object, such that we no longer need to repeat the same multi-threading
code in all gxf::System implementation classes, like All Schedulers and GraphWorker/GraphDriver.

This Processing Thread abstracts and unifies all common parts and behaviors required to implement an efficient and thread-safe gxf::System class, eg async_run_abi(), wait_abi(), stop_abi(), etc.

## Scalability
It can be easily extended multiple threads by either:

Outside: grouping them into a bigger object; or

Inside: replace the single thread with a thread pool

## This implementation provides solid thread safety practices:

## QueueThread Construction:
When a QueueThread is constructed, a new thread is started, and the thread name is set with a limit of 15 characters, as per POSIX pthread_setname_np limitations.

## Stopping Mechanism:
The stop() method sets a stop request flag;
wakes up any waiting threads;
notifies any waiting on the condition variable;
and then attempts to join the thread.

## Waiting Mechanism:
The wait() method blocks until stop() has been called and the queue size is zero, indicating that all items have been processed. It then attempts to join the thread.

## Joining Mechanism:
The joinThread() method is protected by a mutex to prevent multiple threads from calling join() on the same thread object simultaneously.

## Destructor:
The destructor checks if the thread has been joined before attempting to stop and join the thread, ensuring that the resources are cleaned up properly.

## Thread Loop:
The threadLoop() continuously processes items from the queue until a stop request is detected and the queue is empty, after which it clears the queue and notifies that it's exiting.

## Wakeup and Unexpected Exceptions:
Custom exceptions are well-defined and used to manage the flow inside the GuardQueue and QueueThread.

## GuardQueue Class:
The GuardQueue class manages the thread-safe queue with condition variables to notify when data is available or when a wakeup is requested.

## Potential issues:
The only potential issue is if m_StopRequested is set to true while items are still being pushed to the queue, there could be a race condition where items are added after m_StopRequested has been checked but before the queue is cleared. To avoid losing any items, please clear the queue only after confirming that no more items will be pushed, which typically would be managed by the external logic controlling the QueueThread.

## Notes:
### Notification in stop() and threadLoop():

m_StopCond.notify_all() in the stop() is to wake up any threads that are waiting in the wait().
These waiting threads are blocked until stop() is called, after which they should be released to proceed.

The notification at the end of threadLoop() is to handle the case where wait() might be called after stop() has already finished.
If a thread enters wait() at this point, it could potentially wait indefinitely because stop() won't be called again to notify it. The additional notification ensures that any threads entering wait() after stop() has finished will not block indefinitely.

### Purpose of wakeupOnce():

The wakeupOnce() method is used to interrupt the blocking call to pop() on the queue.
When wakeupOnce() is called, it sets a flag (m_WakeupOnce) and then notifies all waiting threads. When pop() is invoked the next time, it will see that m_WakeupOnce is set and will return empty Entry with stop signal.

This mechanism is useful for stopping the QueueThread gracefully. When stop() is called, it invokes wakeupOnce() to ensure that if the thread is currently waiting for an item to be pushed onto the queue, it will stop waiting, send stop Entry, and then exit the thread loop. This is a common pattern for interrupting threads that are blocked waiting for more work.
