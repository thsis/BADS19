#!/usr/bin/env bash
tail -n 250 logs/random_forest.log | grep -o -P '(?<=[0-9\s]\.\s)(.*?)(\s)' | head -n 20
