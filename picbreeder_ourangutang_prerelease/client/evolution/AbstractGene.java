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

package client.evolution;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import client.utilities.XML;

/**
 * AbstractGene provides basic functionality for any gene that may
 * occur in the system.
 *
 * @author Nick Beato
 */

public abstract class AbstractGene implements Gene {
	/**
	 * The marking of the gene.
	 */
	private final Marking marking;
	
	/**
	 * Creates a gene with an invalid marking.  The invalid marking
	 * should be correctly updated by loading it from a file.
	 */
	protected AbstractGene() {
		this.marking = GeneticFactoryInstance.get().createInvalidMarking();
	}
	
	/**
	 * Creates a gene with the specified marking.
	 * 
	 * @param marking The evolutionary identifier
	 */
	protected AbstractGene(Marking marking) {
		this.marking = marking;
	}
	
	/*
	 * (non-Javadoc)
	 * @see client.evolution.Gene#getMarking()
	 */
	public final Marking getMarking() {
		return marking;
	}
	
	/*
	 * (non-Javadoc)
	 * @see client.evolution.Gene#matches(client.evolution.Marking)
	 */
	public final boolean matches(Marking otherMarking) {
		return marking.compareTo(otherMarking) == 0;
	}
	
	/*
	 * (non-Javadoc)
	 * @see java.lang.Comparable#compareTo(java.lang.Object)
	 */
	public int compareTo(Gene otherGene) {
		return marking.compareTo(otherGene.getMarking());
	}

	/*
	 * (non-Javadoc)
	 * @see client.Transferable#load(org.w3c.dom.Element)
	 */
	public void load(Element xmlElement) {
		XML.loadElement(marking, xmlElement);
	}
	
	/*
	 * (non-Javadoc)
	 * @see client.Transferable#store(org.w3c.dom.Element, org.w3c.dom.Document)
	 */
	public void store(Element xmlElement, Document xmlDocument) {
		XML.storeElement(marking, xmlElement, xmlDocument);
	}
}
