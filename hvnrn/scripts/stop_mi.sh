#!/bin/sh

ps aux | grep "python3 mi.py" | grep -v grep | awk '{ print "kill -9", $2 }' | sh
