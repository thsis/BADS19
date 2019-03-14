#!/usr/bin/env bash
tail -n 300 logs/random_forest.log | grep -o -P '(?<=[0-9\s]\.\s)(.*?)(\s)' | head -n 61
