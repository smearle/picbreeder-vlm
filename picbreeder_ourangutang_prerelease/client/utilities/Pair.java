/*
 * Unlicensed intellectual property of the University of Central Florida for
 * internal usage only. You may not distribute this code to anyone. You may
 * not use this code (as source or compiled) or information obtained from
 * this code without permission.
 *
 * Picbreeder Project
 * Evolutionary Complexity Research Group
 * School of Electrical Engineering and Computer Science
 * 2006-2007
 */

package client.utilities;

public final class Pair<E extends Comparable <E> , T extends Comparable <T> > implements Comparable<Pair<E, T> > {
	private final E first;
	private final T second;
	
	public Pair(E first, T second) {
		this.first = first;
		this.second = second;
	}
	
	public int compareTo(Pair<E,T> other) {
		int t = first.compareTo(other.first);
		if(t != 0)
			return t;
		else
			return second.compareTo(other.second);
	}
	
	public E getFirst() {
		return first;
	}
	
	public T getSecond() {
		return second;
	}
}
